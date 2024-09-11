import random
import sys

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Factors taken from ComfyUI:
# https://github.com/comfyanonymous/ComfyUI/blob/81778a7feb233ccba046555d1cc90e7455df93ea/comfy/latent_formats.py#L112
LATENT_RGB_FACTORS = [
    [-0.0645, 0.0177, 0.1052],
    [0.0028, 0.0312, 0.0650],
    [0.1848, 0.0762, 0.0360],
    [0.0944, 0.0360, 0.0889],
    [0.0897, 0.0506, -0.0364],
    [-0.0020, 0.1203, 0.0284],
    [0.0855, 0.0118, 0.0283],
    [-0.0539, 0.0658, 0.1047],
    [-0.0057, 0.0116, 0.0700],
    [-0.0412, 0.0281, -0.0039],
    [0.1106, 0.1171, 0.1220],
    [-0.0248, 0.0682, -0.0481],
    [0.0815, 0.0846, 0.1207],
    [-0.0120, -0.0055, -0.0867],
    [-0.0749, -0.0634, -0.0456],
    [-0.1418, -0.1457, -0.1259],
]


class DiffusersThread(QThread):
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image)

    def __init__(self):
        super().__init__()

        self.pipe = None
        self.prompt = None

    def run(self):
        if self.pipe is None:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")
            self.pipe.enable_model_cpu_offload()

        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        image = self.pipe(
            self.prompt,
            guidance_scale=4.5,
            num_inference_steps=28,
            generator=generator,
            callback_on_step_end=self.step_progress_update,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        self.preview_image(image)

    def step_progress_update(self, _pipe, step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        self.progress_update.emit(step, latents)
        return callback_kwargs

    def preview_image(self, image):
        self.generation_finished.emit(image)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Diffusers Preview")
        self.setMinimumSize(550, 650)

        self.thread = DiffusersThread()
        self.thread.progress_update.connect(self.step_progress_update)
        self.thread.generation_finished.connect(self.generation_finished)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 5, 10, 5)
        main_layout.setSpacing(10)

        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_preview)

        self.prompt_input = QTextEdit()
        main_layout.addWidget(self.prompt_input)

        generate_button = QPushButton("Generate")
        generate_button.clicked.connect(self.on_generate)
        main_layout.addWidget(generate_button)

        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 1)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def on_generate(self):
        self.thread.prompt = self.prompt_input.toPlainText()
        self.thread.start()

    def step_progress_update(self, step: int, latents: torch.Tensor):
        latent_rgb_factors = torch.tensor(LATENT_RGB_FACTORS, dtype=latents.dtype).to(
            device=latents.device
        )

        latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
        latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
        image = Image.fromarray(latents_ubyte.byte().cpu().numpy())

        self.show_preview(image)

    def generation_finished(self, image: Image):
        self.show_preview(image)

    def show_preview(self, image: Image):
        qimage = QImage(
            image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888
        )
        qpixmap = QPixmap.fromImage(qimage)

        label_size = self.image_preview.size()

        scaled_pixmap = qpixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_preview.setPixmap(scaled_pixmap)


class DiffusersPreviewApp(QApplication):
    def __init__(self, argv):
        super(DiffusersPreviewApp, self).__init__(argv)

        self.window = MainWindow()
        self.window.show()


def main():
    app = DiffusersPreviewApp(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
