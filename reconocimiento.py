#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from message_filters import Subscriber as MFSub, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

import torch

cv2.setNumThreads(1)  # evita sobre-subprocesos de OpenCV que compiten con PyTorch


def _import_ultralytics_or_die(node: Node):
    try:
        from ultralytics import YOLO  # noqa: F401
        return YOLO
    except Exception as e:
        node.get_logger().error(
            "No se pudo importar 'ultralytics'.\n"
            "Solución rápida (en el mismo entorno Python que usa ROS 2):\n"
            "  pip3 install ultralytics\n"
            "NOTA: Activa tu workspace antes: `source install/setup.bash`.\n"
            f"Error original: {e}"
        )
        raise


class ReconD435Node(Node):
    def __init__(self):
        super().__init__("recon_d435_3d_fast")

        # ---------- Parámetros ----------
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("output_image_topic", "/reconocimiento/image")
        self.declare_parameter("output_point_topic", "/reconocimiento/point")
        self.declare_parameter("model_path", "/home/kruger/ros2_ws1/src/cajas_description/algoritmo/best.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.10)
        self.declare_parameter("iou", 0.50)
        self.declare_parameter("use_rgb", False)
        self.declare_parameter("target_frame", "")
        self.declare_parameter("sample_step_px", 4)
        self.declare_parameter("depth_scale_if_16UC1", 0.001)
        self.declare_parameter("draw_corners", True)

        # Robustez extra
        self.declare_parameter("erode_kernel_px", 7)
        self.declare_parameter("depth_win_px", 11)
        self.declare_parameter("corner_min_dist_px", 6)
        self.declare_parameter("corner_max_shift_px", 16)

        # Rendimiento extra
        self.declare_parameter("max_infer_hz", 0.0)   # 0 = sin límite; ej: 20.0 para cap
        self.declare_parameter("skip_if_busy", True)  # dropea frame si la red sigue ocupada
        self.declare_parameter("use_half", True)      # FP16 si hay CUDA

        # ---------- Asignación ----------
        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.info_topic = self.get_parameter("camera_info_topic").value
        self.output_image_topic = self.get_parameter("output_image_topic").value
        self.output_point_topic = self.get_parameter("output_point_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.use_rgb = bool(self.get_parameter("use_rgb").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.sample_step = int(self.get_parameter("sample_step_px").value)
        self.depth_scale_16u = float(self.get_parameter("depth_scale_if_16UC1").value)
        self.draw_corners = bool(self.get_parameter("draw_corners").value)

        self.erode_k = int(self.get_parameter("erode_kernel_px").value)
        self.depth_win = int(self.get_parameter("depth_win_px").value)
        self.corner_min_dist = int(self.get_parameter("corner_min_dist_px").value)
        self.corner_max_shift = int(self.get_parameter("corner_max_shift_px").value)

        self.max_infer_hz = float(self.get_parameter("max_infer_hz").value)
        self.skip_if_busy = bool(self.get_parameter("skip_if_busy").value)
        self.use_half = bool(self.get_parameter("use_half").value)

        # ---------- YOLO ----------
        YOLO = _import_ultralytics_or_die(self)
        self.model_lock = threading.Lock()
        try:
            self.modelo = YOLO(self.model_path)
            try:
                self.modelo.fuse()
            except Exception:
                pass
            self.get_logger().info(f"YOLO cargado: {self.model_path}")
        except Exception as e:
            self.get_logger().fatal(f"No se pudo cargar el modelo: {e}")
            raise

        # Dispositivo y precisión
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.use_half and (self.device == "cuda")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.get_logger().info(f"Dispositivo: {self.device} | half={self.half}")

        # Warm-up seguro (solo autocast si CUDA+half)
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            if self.half and self.device == "cuda":
                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _ = self.modelo.predict(
                        dummy,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        half=False,
                        retina_masks=True,
                    )
            else:
                with torch.inference_mode():
                    _ = self.modelo.predict(
                        dummy,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        half=False,
                        retina_masks=True,
                    )
        except Exception:
            pass

        # ---------- Intrínsecos ----------
        self.have_info = False
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_frame = None
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)

        # ---------- TF ----------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---------- ROS IO ----------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.bridge = CvBridge()
        self.pub_img = self.create_publisher(Image, self.output_image_topic, qos)
        self.pub_point = self.create_publisher(PointStamped, self.output_point_topic, qos)

        # Sincronización color + depth
        self.sub_color = MFSub(self, Image, self.color_topic, qos_profile=qos)
        self.sub_depth = MFSub(self, Image, self.depth_topic, qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([self.sub_color, self.sub_depth], queue_size=5, slop=0.03)
        self.sync.registerCallback(self.sync_cb)

        # Kernels prealocados
        ksz = max(1, self.erode_k | 1)
        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

        # Control de tasa
        self._last_infer_ts = 0.0
        self._infer_busy = False

        self.get_logger().info(
            f"Color: {self.color_topic} | Depth(aligned): {self.depth_topic} | CameraInfo: {self.info_topic}"
        )

    # ---------- CameraInfo ----------
    def info_cb(self, msg: CameraInfo):
        try:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
            self.camera_frame = msg.header.frame_id
            self.have_info = True
            self.destroy_subscription(self.info_sub)
            self.get_logger().info(
                f"Intrínsecos OK: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f} | frame={self.camera_frame}"
            )
        except Exception as e:
            self.get_logger().warn(f"No se pudo leer CameraInfo: {e}")

    # ---------- Utilidades ----------
    def _depth_to_meters(self, depth_img, encoding):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) * self.depth_scale_16u
        return depth_img.astype(np.float32)

    def _project_uvz_to_xyz(self, u, v, z_m):
        X = (u - self.cx) * z_m / self.fx
        Y = (v - self.cy) * z_m / self.fy
        return float(X), float(Y), float(z_m)

    def _median_depth_in_mask(self, depth_m, mask, step=4):
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        if step > 1:
            ys = ys[::step]
            xs = xs[::step]
        vals = depth_m[ys, xs]
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > 0.05) & (vals < 10.0)]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _robust_depth_at_pixel(self, depth_m, u, v, win=5):
        h, w = depth_m.shape
        u = int(u)
        v = int(v)
        r = max(1, win // 2)
        x0 = max(0, u - r)
        x1 = min(w, u + r + 1)
        y0 = max(0, v - r)
        y1 = min(h, v + r + 1)
        roi = depth_m[y0:y1, x0:x1].reshape(-1)
        roi = roi[np.isfinite(roi)]
        roi = roi[(roi > 0.05) & (roi < 10.0)]
        if roi.size == 0:
            return None
        return float(np.median(roi))

    def _pull_inside(self, u, v, uc, vc, mask_bin, max_px=16, min_dist=6):
        """Empuja (u,v) hacia el centro (uc,vc) como máximo max_px, y clamp a 'min_dist' del borde."""
        h, w = mask_bin.shape
        u = int(np.clip(u, min_dist, w - 1 - min_dist))
        v = int(np.clip(v, min_dist, h - 1 - min_dist))
        if mask_bin[v, u]:
            return u, v
        du = np.sign(uc - u)
        dv = np.sign(vc - v)
        uu, vv = u, v
        for _ in range(max_px):
            uu = int(np.clip(uu + du, min_dist, w - 1 - min_dist))
            vv = int(np.clip(vv + dv, min_dist, h - 1 - min_dist))
            if mask_bin[vv, uu]:
                return uu, vv
        return u, v

    def _interp_points_between(self, p0, p1, n=3):
        """Devuelve n puntos interpolados linealmente ENTRE p0 y p1 (excluye extremos).
        p0, p1: (u,v) en pixeles.
        """
        if n <= 0:
            return []
        u0, v0 = float(p0[0]), float(p0[1])
        u1, v1 = float(p1[0]), float(p1[1])
        pts = []
        for k in range(1, n + 1):
            t = k / (n + 1.0)  # 0<t<1
            pts.append((u0 + (u1 - u0) * t, v0 + (v1 - v0) * t))
        return pts

    def _draw_label(self, frame, u, v, text, color_bgr, dy=-10):
        """Texto pequeño, del mismo color que el punto."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45  # tamaño (baja a 0.40 si aún lo ves grande)
        thick = 1
        x = int(u) + 6
        y = int(v) + dy

        # borde negro fino para contraste (mantiene el color principal)
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color_bgr, thick, cv2.LINE_AA)

    # ---------- Pipeline principal ----------
    def sync_cb(self, color_msg: Image, depth_msg: Image):
        if not self.have_info:
            return

        # Gate de FPS e inferencia ocupada
        now = time.time()
        if self.max_infer_hz > 0.0 and (now - self._last_infer_ts) < (1.0 / self.max_infer_hz):
            return
        if self.skip_if_busy and self._infer_busy:
            return
        self._infer_busy = True
        self._last_infer_ts = now

        try:
            # 1) Conversión imágenes
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg)
            depth_m = self._depth_to_meters(depth_raw, depth_msg.encoding)

            # Asegurar tamaños iguales
            if color.shape[:2] != depth_m.shape[:2]:
                H = min(color.shape[0], depth_m.shape[0])
                W = min(color.shape[1], depth_m.shape[1])
                color = color[:H, :W]
                depth_m = depth_m[:H, :W]

            # 2) YOLO (autocast solo si CUDA+half)
            frame_in = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) if self.use_rgb else color

            if self.half and self.device == "cuda":
                with self.model_lock, torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    results = self.modelo.predict(
                        frame_in,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        half=False,          # FP16 lo maneja autocast
                        retina_masks=True,
                    )
            else:
                with self.model_lock, torch.inference_mode():
                    results = self.modelo.predict(
                        frame_in,
                        imgsz=self.imgsz,
                        conf=self.conf,
                        iou=self.iou,
                        device=self.device,
                        verbose=False,
                        half=False,
                        retina_masks=True,
                    )

            frame = color  # overlay directo

            # 3) Mejor máscara (por score) + minAreaRect
            mask = None
            box_pts = None
            best_score = -1.0

            if len(results) > 0 and hasattr(results[0], "masks") and results[0].masks is not None:
                res = results[0]
                masks = (res.masks.data.detach().cpu().numpy() > 0.5)  # (N,H,W)
                scores = None
                if hasattr(res, "boxes") and res.boxes is not None and hasattr(res.boxes, "conf"):
                    scores = res.boxes.conf.detach().cpu().numpy()

                N = masks.shape[0]
                for i in range(N):
                    m = masks[i]
                    if m.shape != (frame.shape[0], frame.shape[1]):
                        m = cv2.resize(
                            m.astype(np.uint8),
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    sc = scores[i] if scores is not None else 1.0
                    if sc > best_score:
                        best_score = sc
                        mask = (m.astype(np.uint8) * 255)

                if mask is not None:
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        rect = cv2.minAreaRect(cnt)
                        box_pts = cv2.boxPoints(rect).astype(np.int32)

            # Fallback a bbox si no hubo máscara
            if mask is None and len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
                res = results[0]
                if len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.detach().cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    j = int(np.argmax(areas))
                    x1, y1, x2, y2 = xyxy[j].astype(int)
                    mask = np.zeros(frame.shape[:2], np.uint8)
                    mask[y1:y2, x1:x2] = 255
                    box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

            if mask is None:
                self._publish_image(frame, color_msg.header)
                return

            # 4) Núcleo rápido (erosión) + mediana global
            if self.erode_k > 1:
                mask_in = cv2.erode(mask, self.kernel_erode, iterations=1)
                if np.count_nonzero(mask_in) < 50:
                    mask_in = mask
            else:
                mask_in = mask

            z_med = self._median_depth_in_mask(depth_m, mask_in > 0, step=self.sample_step)
            if z_med is None:
                self._publish_image(frame, color_msg.header)
                return

            # 5) Centroide 2D
            M = cv2.moments(mask_in, binaryImage=True)
            if M["m00"] > 0:
                cx2d = int(M["m10"] / M["m00"])
                cy2d = int(M["m01"] / M["m00"])
            else:
                cx2d = int(np.mean(box_pts[:, 0]))
                cy2d = int(np.mean(box_pts[:, 1]))

            zc = self._robust_depth_at_pixel(depth_m, cx2d, cy2d, win=self.depth_win) or z_med
            Xc, Yc, Zc = self._project_uvz_to_xyz(cx2d, cy2d, zc)

            # 6) Esquinas + (puntos interpolados), pero SOLO centroide y esquinas con XYZ
            if self.draw_corners and box_pts is not None:
                mask_bin = (mask_in > 0).astype(np.uint8)

                # Ajustar y dibujar 4 esquinas (AZUL) + texto XYZ
                corners_uv = []
                for (u, v) in box_pts:
                    u_in, v_in = self._pull_inside(
                        u, v, cx2d, cy2d, mask_bin,
                        max_px=self.corner_max_shift, min_dist=self.corner_min_dist
                    )
                    corners_uv.append((float(u_in), float(v_in)))

                    zk = self._robust_depth_at_pixel(depth_m, u_in, v_in, win=self.depth_win) or z_med
                    Xk, Yk, Zk = self._project_uvz_to_xyz(u_in, v_in, zk)

                    cv2.circle(frame, (int(u_in), int(v_in)), 4, (255, 0, 0), -1)
                    txt_k = f"X:{Xk:.2f} Y:{Yk:.2f} Z:{Zk:.2f}m"
                    self._draw_label(frame, u_in, v_in, txt_k, (255, 0, 0), dy=-10)

                # Interpolación: 3 puntos EXTRA por arista (VERDE) SIN texto
                if len(corners_uv) == 4:
                    for i in range(4):
                        p0 = corners_uv[i]
                        p1 = corners_uv[(i + 1) % 4]
                        edge_pts = self._interp_points_between(p0, p1, n=3)

                        for (uu, vv) in edge_pts:
                            uu_in, vv_in = self._pull_inside(
                                uu, vv, cx2d, cy2d, mask_bin,
                                max_px=self.corner_max_shift, min_dist=self.corner_min_dist
                            )
                            cv2.circle(frame, (int(uu_in), int(vv_in)), 3, (0, 255, 0), -1)

            # 7) Dibujo del centroide (ROJO) + texto XYZ
            cv2.circle(frame, (cx2d, cy2d), 5, (0, 0, 255), -1)
            txt_c = f"X:{Xc:.2f} Y:{Yc:.2f} Z:{Zc:.2f}m"
            self._draw_label(frame, cx2d, cy2d, txt_c, (0, 0, 255), dy=-12)

            # 8) Publicar PointStamped (centroide) (y TF opcional)
            pt_msg = PointStamped()
            pt_msg.header = color_msg.header
            if self.camera_frame:
                pt_msg.header.frame_id = self.camera_frame
            pt_msg.point.x, pt_msg.point.y, pt_msg.point.z = Xc, Yc, Zc

            if self.target_frame:
                try:
                    tf = self.tf_buffer.lookup_transform(
                        self.target_frame, pt_msg.header.frame_id, rclpy.time.Time()
                    )
                    pt_out = do_transform_point(pt_msg, tf)
                    self.pub_point.publish(pt_out)
                except (LookupException, ConnectivityException, ExtrapolationException):
                    self.pub_point.publish(pt_msg)
            else:
                self.pub_point.publish(pt_msg)

            # 9) Publicar imagen overlay
            self._publish_image(frame, color_msg.header)

        except Exception as e:
            self.get_logger().warn(f"Pipeline fallo: {e}")
        finally:
            self._infer_busy = False

    def _publish_image(self, frame_bgr, header):
        try:
            msg_out = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
            msg_out.header = header
            if self.camera_frame:
                msg_out.header.frame_id = self.camera_frame
            self.pub_img.publish(msg_out)
        except Exception as e:
            self.get_logger().warn(f"No se pudo publicar la imagen: {e}")


def main():
    rclpy.init()
    node = ReconD435Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
