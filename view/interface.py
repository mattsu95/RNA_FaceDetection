"""
FRM Estimativa de idade por detecção facial
Interface desktop em Python (CustomTkinter).

Equipe: Pedro Miotto, Eduardo Nogueira Korte, Thiago Oliveira, Matheus Seghatti
Modelo: Rede Neural Convolucional

Como executar:
    pip install -r requirements.txt
    python interface-frm.py

    A logo (image.png) precisa estar na mesma pasta deste arquivo.

Esta versão contém somente a interface. A integração do modelo será feita
posteriormente em uma camada separada.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "image.png")
SRC_DIR = os.path.join(os.path.dirname(BASE_DIR), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FaceDetection import FaceDetection

BG = "#ffffff"
PANEL = "#f5f9fc"
PANEL_2 = "#e4eef7"
LINE = "#a9c3d8"
LINE_SOFT = "#c9d9e6"
CYAN = "#007f80"
BLUE = "#246aa3"
BLUE_DIM = "#4f7ea7"
CREAM = "#102a43"
MUTED = "#3f596f"
MUTED_DIM = "#536b7f"

# botões sólidos, no estilo da referência (Face Recognition System)
BTN_BLUE = "#2f5fa8"
BTN_BLUE_HOVER = "#3d6fba"

FONT_DISPLAY = "Segoe UI"     # título / número de idade
FONT_BODY = "Segoe UI"        # sans — texto de interface
FONT_MONO = "Consolas"        # mono — leituras técnicas

ctk.set_appearance_mode("light")


def font(family, size, weight="normal", slant="roman"):
    return ctk.CTkFont(family=family, size=size, weight=weight, slant=slant)


# ---------------------------------------------------------------------------
# Tela 1 — Introdução
# ---------------------------------------------------------------------------
class IntroFrame(ctk.CTkFrame):
    def __init__(self, master, on_start):
        super().__init__(master, fg_color=BG)

        wrapper = ctk.CTkFrame(self, fg_color="transparent")
        wrapper.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            wrapper, text="FRM", text_color=CREAM,
            font=font(FONT_DISPLAY, 56, weight="bold"),
        ).pack(pady=(0, 4))

        ctk.CTkLabel(
            wrapper, text="Face Recognition Module",
            text_color=BLUE, font=font(FONT_MONO, 16),
        ).pack(pady=(0, 14))

        ctk.CTkLabel(
            wrapper, text="Estimativa de idade com detecção facial",
            text_color=CREAM, font=font(FONT_DISPLAY, 19),
            wraplength=420, justify="center",
        ).pack(pady=(0, 30))


        start_btn = ctk.CTkButton(
            wrapper, text="Iniciar", width=220, height=44, corner_radius=6,
            fg_color=BTN_BLUE, hover_color=BTN_BLUE_HOVER, border_width=0,
            text_color="white",
            font=font(FONT_BODY, 14, weight="bold"),
            command=on_start,
        )
        start_btn.pack()

class Viewport(ctk.CTkFrame):
    CANVAS_W, CANVAS_H = 420, 500
    CORNER = 26

    def __init__(self, master):
        super().__init__(master, fg_color=PANEL, corner_radius=4,
                          border_width=1, border_color=LINE_SOFT)
        self.canvas = tk.Canvas(
            self, width=self.CANVAS_W, height=self.CANVAS_H,
            bg=PANEL, highlightthickness=0,
        )
        self.canvas.pack(padx=1, pady=1)

        self.current_image = None       # PIL.Image atualmente carregada
        self._tk_image = None           # referência viva do PhotoImage
        self._draw_empty_state()

    # -- desenho base -----------------------------------------------------
    def _clear(self):
        self.canvas.delete("all")

    def _draw_corners(self):
        c, w, h = self.CORNER, self.CANVAS_W, self.CANVAS_H
        pad = 14
        for (x, y, dx, dy) in [
            (pad, pad, 1, 1), (w - pad, pad, -1, 1),
            (pad, h - pad, 1, -1), (w - pad, h - pad, -1, -1),
        ]:
            self.canvas.create_line(x, y, x + dx * c, y, fill=BLUE, width=2)
            self.canvas.create_line(x, y, x, y + dy * c, fill=BLUE, width=2)

    def _draw_empty_state(self):
        self._clear()
        self.canvas.create_rectangle(0, 0, self.CANVAS_W, self.CANVAS_H, fill=PANEL, outline="")
        self.canvas.create_text(
            self.CANVAS_W / 2, self.CANVAS_H / 2 - 10,
            text="Nenhuma imagem carregada", fill=MUTED_DIM,
            font=(FONT_BODY, 13),
        )
        self.canvas.create_text(
            self.CANVAS_W / 2, self.CANVAS_H / 2 + 16,
            text="Envie uma foto para visualizar a prévia", fill=MUTED_DIM,
            font=(FONT_BODY, 13),
        )
        self._draw_corners()

    # -- exibição de imagem estática ---------------------------------------
    def show_image(self, pil_img):
        self.current_image = pil_img
        self._render_cover(pil_img)
        self._draw_corners()

    def show_bgr_frame(self, bgr_frame):
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self.show_image(Image.fromarray(rgb_frame))

    def _render_cover(self, pil_img, clear=True):
        """Redimensiona a imagem para preencher o canvas (comportamento tipo 'cover')."""
        if clear:
            self._clear()
            self.canvas.create_rectangle(0, 0, self.CANVAS_W, self.CANVAS_H, fill=PANEL, outline="")
        fitted = ImageOps.fit(pil_img, (self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(fitted)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

    # -- reset -----------------------------------------------------------
    def reset(self):
        self.current_image = None
        self._draw_empty_state()


# ---------------------------------------------------------------------------
# Painel de resultado
# ---------------------------------------------------------------------------
class ResultPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL, corner_radius=4,
                          border_width=1, border_color=LINE_SOFT)

        ctk.CTkLabel(
            self, text="RESULTADO", text_color=BLUE,
            font=font(FONT_MONO, 11, weight="bold"), anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 14))

        self.empty_label = ctk.CTkLabel(
            self, text="Aguardando imagem para estimar a idade.",
            text_color=MUTED, font=font(FONT_BODY, 12),
            anchor="w", justify="left",
        )
        self.empty_label.pack(fill="x", padx=20)

        self.body = ctk.CTkFrame(self, fg_color="transparent")

        age_row = ctk.CTkFrame(self.body, fg_color="transparent")
        age_row.pack(fill="x", anchor="w")
        self.age_number = ctk.CTkLabel(
            age_row, text="—", text_color=CREAM,
            font=font(FONT_DISPLAY, 46, weight="bold"),
        )
        self.age_number.pack(side="left")
        ctk.CTkLabel(
            age_row, text=" rostos detectados", text_color=MUTED,
            font=font(FONT_BODY, 14),
        ).pack(side="left", padx=(6, 0), pady=(14, 0))

        self.age_range = ctk.CTkLabel(
            self.body, text="", text_color=MUTED,
            font=font(FONT_BODY, 12), anchor="w",
        )
        self.age_range.pack(fill="x", pady=(2, 18))

        conf_row = ctk.CTkFrame(self.body, fg_color="transparent")
        conf_row.pack(fill="x")
        ctk.CTkLabel(
            conf_row, text="CONFIANÇA", text_color=MUTED,
            font=font(FONT_MONO, 10, weight="bold"),
        ).pack(side="left")
        self.confidence_value = ctk.CTkLabel(
            conf_row, text="0%", text_color=CREAM, font=font(FONT_BODY, 12, weight="bold"),
        )
        self.confidence_value.pack(side="right")

        self.progress = ctk.CTkProgressBar(
            self.body, height=4, corner_radius=2,
            fg_color=LINE_SOFT, progress_color=CYAN,
        )
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(6, 20))

        ctk.CTkLabel(
            self.body, text="●  Prévia da interface",
            text_color=MUTED, font=font(FONT_BODY, 11),
            fg_color=BG, corner_radius=14,
        ).pack(anchor="w", ipadx=10, ipady=5)

        self.reset_btn = ctk.CTkButton(
            self, text="Nova análise", fg_color="transparent",
            hover_color=PANEL_2, text_color=BLUE, font=font(FONT_BODY, 12, weight="bold"),
            anchor="w", width=1, height=20,
        )

    def show_face_count(self, face_count):
        self.empty_label.pack_forget()
        self.body.pack(fill="x", padx=20, pady=(0, 16))
        self.reset_btn.pack(fill="x", padx=20, pady=(0, 14))
        self.age_number.configure(text=str(face_count))
        self.age_range.configure(
            text="rosto detectado" if face_count == 1 else "rostos detectados"
        )
        self.confidence_value.configure(text="Detecção facial")
        self.progress.set(1 if face_count else 0)

    def reset(self):
        self.body.pack_forget()
        self.reset_btn.pack_forget()
        self.empty_label.configure(text="Aguardando imagem para estimar a idade.")
        self.empty_label.pack(fill="x", padx=20)
        self.age_number.configure(text="—")
        self.age_range.configure(text="")
        self.confidence_value.configure(text="0%")
        self.progress.set(0)


# ---------------------------------------------------------------------------
# Tela 2 — Captura e resultado
# ---------------------------------------------------------------------------
class AppFrame(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color=BG)
        self.detector = None
        self.camera_running = False
        self.camera_job = None

        # cabeçalho ------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(22, 18))

        ctk.CTkButton(
            header, text="‹  Voltar", fg_color="transparent",
            hover_color=PANEL, text_color=MUTED, font=font(FONT_BODY, 13),
            width=70, command=self.go_back,
        ).pack(side="left")

        ctk.CTkLabel(
            header, text="FRM", text_color=CREAM,
            font=font(FONT_DISPLAY, 20, weight="bold"),
        ).pack(side="left", expand=True)

        ctk.CTkLabel(
            header, text="●  Modelo em treinamento", text_color=BLUE,
            font=font(FONT_BODY, 11, weight="bold"), fg_color=PANEL, corner_radius=14,
        ).pack(side="right", ipadx=10, ipady=5)

        ctk.CTkFrame(self, fg_color=LINE_SOFT, height=1).pack(fill="x", padx=28)

        # corpo ------------------------------------------------------------
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28, pady=22)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", padx=(0, 24))

        self.viewport = Viewport(left)
        self.viewport.pack()
        right = ctk.CTkFrame(body, fg_color="transparent", width=300)
        right.pack(side="left", fill="both", expand=True)

        actions = ctk.CTkFrame(right, fg_color="transparent")
        actions.pack(fill="x", pady=(0, 18))
        ctk.CTkButton(
            actions, text="Enviar foto", fg_color=BTN_BLUE, hover_color=BTN_BLUE_HOVER,
            border_width=0, text_color="white",
            font=font(FONT_BODY, 13, weight="bold"), height=44, corner_radius=6,
            command=self.upload_photo,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.webcam_btn = ctk.CTkButton(
            actions, text="Ligar webcam", fg_color=PANEL_2, hover_color=LINE_SOFT,
            border_width=1, border_color=LINE, text_color=BLUE,
            font=font(FONT_BODY, 13, weight="bold"), height=44, corner_radius=6,
            command=self.toggle_webcam,
        )
        self.webcam_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.webcam_status = ctk.CTkLabel(
            right, text="", text_color=MUTED, font=font(FONT_BODY, 11),
            wraplength=300, justify="left", anchor="w",
        )
        self.webcam_status.pack(fill="x", pady=(0, 14))
        self.result_panel = ResultPanel(right)
        self.result_panel.pack(fill="x")
        self.result_panel.reset_btn.configure(command=self.reset_all)

        self.on_back = on_back

    def go_back(self):
        self.stop_webcam()
        self.on_back()

    # -- ações -------------------------------------------------------------
    def upload_photo(self):
        path = filedialog.askopenfilename(
            title="Selecionar foto",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        try:
            if self.camera_running:
                self.stop_webcam()
            detector = self.get_detector()
            bgr_image = cv2.imread(path)
            if bgr_image is None:
                raise ValueError("Não foi possível carregar a imagem selecionada.")
            result_frame, face_count = detector.process_frame(bgr_image)
            self.viewport.show_bgr_frame(result_frame)
            self.result_panel.show_face_count(face_count)
            self.webcam_status.configure(text="Imagem processada pelo detector facial.")
        except Exception as error:
            self.webcam_status.configure(text=f"Erro ao processar a imagem: {error}")

    def get_detector(self):
        if self.detector is None:
            self.webcam_status.configure(text="Carregando modelo de detecção facial...")
            self.update_idletasks()
            self.detector = FaceDetection()
        return self.detector

    def toggle_webcam(self):
        if self.camera_running:
            self.stop_webcam()
            return
        try:
            detector = self.get_detector()
            detector.open_camera()
            self.camera_running = True
            self.webcam_btn.configure(text="Desligar webcam")
            self.webcam_status.configure(text="Webcam ativa. Exibindo detecção em tempo real.")
            self.update_webcam_frame()
        except Exception as error:
            self.webcam_status.configure(text=f"Não foi possível iniciar a webcam: {error}")

    def update_webcam_frame(self):
        if not self.camera_running:
            return
        try:
            frame, face_count = self.detector.read_camera_frame()
            if frame is None:
                self.stop_webcam()
                self.webcam_status.configure(text="A webcam não retornou nenhum frame.")
                return
            self.viewport.show_bgr_frame(frame)
            self.result_panel.show_face_count(face_count)
            self.camera_job = self.after(15, self.update_webcam_frame)
        except Exception as error:
            self.stop_webcam()
            self.webcam_status.configure(text=f"Erro na webcam: {error}")

    def stop_webcam(self):
        self.camera_running = False
        if self.camera_job is not None:
            self.after_cancel(self.camera_job)
            self.camera_job = None
        if self.detector is not None:
            self.detector.release_camera()
        self.webcam_btn.configure(text="Ligar webcam")

    def reset_all(self):
        self.stop_webcam()
        self.viewport.reset()
        self.result_panel.reset()
        self.webcam_status.configure(text="")

class FRMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FRM — Estimativa de idade")
        self.geometry("980x680")
        self.minsize(860, 620)
        self.configure(fg_color=BG)

        container = ctk.CTkFrame(self, fg_color=BG)
        container.pack(fill="both", expand=True)

        self.intro_frame = IntroFrame(container, on_start=self.show_app)
        self.app_frame = AppFrame(container, on_back=self.show_intro)

        self.intro_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.app_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.show_intro()

    def show_app(self):
        self.intro_frame.lower()
        self.app_frame.lift()

    def show_intro(self):
        self.app_frame.lower()
        self.intro_frame.lift()


if __name__ == "__main__":
    app = FRMApp()
    app.mainloop()