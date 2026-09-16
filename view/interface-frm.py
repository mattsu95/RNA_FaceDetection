"""
FRM (Face Recognition Module) — Estimativa de idade por detecção facial
Interface desktop em Python (CustomTkinter + OpenCV).

Equipe: Pedro Miotto, Eduardo Nogueira Korte, Thiago Oliveira, Matheus Seghatti
Modelo: Rede Neural Convolucional

Como executar:
    pip install -r requirements.txt
    python interface-frm.py

    A logo (image.png) precisa estar na mesma pasta deste arquivo.

PONTO DE INTEGRAÇÃO DO MODELO:
    Procure o método `Viewport.run_analysis` mais abaixo. Ele hoje gera um
    resultado simulado (idade e confiança aleatórias) só para a interface
    ficar completa enquanto o modelo não está pronto. Troque o bloco
    marcado por uma chamada real ao modelo treinado (ex.: carregar o .h5 /
    .pt salvo e rodar a inferência sobre `self.current_image`).
"""

import os
import random
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "image.png")

# ---------------------------------------------------------------------------
# Paleta (derivada da referência de ícone FRM) e tipografia
# ---------------------------------------------------------------------------
BG = "#0a1524"
PANEL = "#0f1f36"
PANEL_2 = "#142c48"
LINE = "#1e3a5a"
LINE_SOFT = "#16293f"
CYAN = "#8ff0e6"
BLUE = "#529ad4"
BLUE_DIM = "#2c5578"
CREAM = "#f2ece0"
MUTED = "#7691aa"
MUTED_DIM = "#4d6478"

# botões sólidos, no estilo da referência (Face Recognition System)
BTN_BLUE = "#2f5fa8"
BTN_BLUE_HOVER = "#3d6fba"

FONT_DISPLAY = "Segoe UI"     # título / número de idade
FONT_BODY = "Segoe UI"        # sans — texto de interface
FONT_MONO = "Consolas"        # mono — leituras técnicas

ctk.set_appearance_mode("dark")


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

        if os.path.exists(LOGO_PATH):
            logo_img = Image.open(LOGO_PATH).convert("RGBA")
            self._logo = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=(96, 96))
            ctk.CTkLabel(wrapper, image=self._logo, text="").pack(pady=(0, 18))

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
            text_color=MUTED, font=font(FONT_DISPLAY, 19),
            wraplength=420, justify="center",
        ).pack(pady=(0, 30))

       
        ctk.CTkLabel(
            wrapper,
            text="Pedro Miotto, Eduardo Nogueira Korte, Thiago Oliveira, Matheus Seghatti",
            text_color=MUTED_DIM, font=font(FONT_MONO, 13),
            wraplength=380, justify="center",
        ).pack(pady=(4, 40))

        start_btn = ctk.CTkButton(
            wrapper, text="Iniciar", width=220, height=44, corner_radius=6,
            fg_color=BTN_BLUE, hover_color=BTN_BLUE_HOVER, border_width=0,
            text_color="white",
            font=font(FONT_BODY, 14, weight="bold"),
            command=on_start,
        )
        start_btn.pack()


# ---------------------------------------------------------------------------
# Viewport — área de captura (imagem / webcam) com molduras de canto
# ---------------------------------------------------------------------------
class Viewport(ctk.CTkFrame):
    CANVAS_W, CANVAS_H = 420, 500
    CORNER = 26

    def __init__(self, master, on_result):
        super().__init__(master, fg_color=PANEL, corner_radius=4,
                          border_width=1, border_color=LINE_SOFT)
        self.on_result = on_result

        self.canvas = tk.Canvas(
            self, width=self.CANVAS_W, height=self.CANVAS_H,
            bg=PANEL, highlightthickness=0,
        )
        self.canvas.pack(padx=1, pady=1)

        self.current_image = None       # PIL.Image atualmente carregada
        self._tk_image = None           # referência viva do PhotoImage
        self._scan_job = None
        self._scan_y = 0
        self._cap = None                # cv2.VideoCapture
        self._webcam_job = None
        self._last_frame = None         # último frame cru da webcam (PIL)

        self.capture_btn = ctk.CTkButton(
            self, text="Capturar", width=110, height=34, corner_radius=6,
            fg_color=BTN_BLUE, hover_color=BTN_BLUE_HOVER, border_width=0,
            text_color="white",
            font=font(FONT_BODY, 12, weight="bold"),
            command=self.capture_from_webcam,
        )
        # posicionado sob o canvas quando a webcam está ativa
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
        self.capture_btn.place_forget()
        self.canvas.create_rectangle(0, 0, self.CANVAS_W, self.CANVAS_H, fill=PANEL, outline="")
        self.canvas.create_text(
            self.CANVAS_W / 2, self.CANVAS_H / 2 - 10,
            text="Nenhuma imagem carregada", fill=MUTED_DIM,
            font=(FONT_MONO, 12),
        )
        self.canvas.create_text(
            self.CANVAS_W / 2, self.CANVAS_H / 2 + 16,
            text="Envie uma foto ou use a webcam", fill=MUTED_DIM,
            font=(FONT_MONO, 12),
        )
        self._draw_corners()

    # -- exibição de imagem estática ---------------------------------------
    def show_image(self, pil_img):
        self.stop_webcam()
        self.current_image = pil_img
        self._render_cover(pil_img)
        self._draw_corners()
        self.run_analysis()

    def _render_cover(self, pil_img, clear=True):
        """Redimensiona a imagem para preencher o canvas (comportamento tipo 'cover')."""
        if clear:
            self._clear()
            self.canvas.create_rectangle(0, 0, self.CANVAS_W, self.CANVAS_H, fill=PANEL, outline="")
        fitted = ImageOps.fit(pil_img, (self.CANVAS_W, self.CANVAS_H), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(fitted)
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

    # -- webcam --------------------------------------------------------------
    def start_webcam(self):
        self.current_image = None
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap = None
            self._draw_empty_state()
            raise RuntimeError("webcam indisponível")
        self.capture_btn.place(relx=0.5, rely=0.94, anchor="s")
        self._poll_webcam()

    def _poll_webcam(self):
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            self._last_frame = pil_frame
            self._render_cover(pil_frame)
            self._draw_corners()
        self._webcam_job = self.after(30, self._poll_webcam)

    def stop_webcam(self):
        if self._webcam_job:
            self.after_cancel(self._webcam_job)
            self._webcam_job = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.capture_btn.place_forget()

    def capture_from_webcam(self):
        if self._last_frame is None:
            return
        snapshot = self._last_frame.copy()
        self.stop_webcam()
        self.current_image = snapshot
        self._render_cover(snapshot)
        self._draw_corners()
        self.run_analysis()

    # -- reset -----------------------------------------------------------
    def reset(self):
        self.stop_webcam()
        self.current_image = None
        self._draw_empty_state()

    # -- análise (varredura visual + resultado simulado) -------------------
    def run_analysis(self):
        self.on_result(None)  # sinaliza "processando" para o painel de resultado
        self._scan_y = 14
        self._animate_scan(frames_left=26)

    def _animate_scan(self, frames_left):
        # redesenha imagem + corners + linha de varredura
        if self.current_image is not None:
            self._render_cover(self.current_image)
        self._draw_corners()
        self.canvas.create_line(
            0, self._scan_y, self.CANVAS_W, self._scan_y, fill=CYAN, width=2,
        )
        self.canvas.create_text(
            16, self.CANVAS_H - 18, anchor="w",
            text="ANALISANDO IMAGEM…", fill=CYAN, font=(FONT_MONO, 10),
        )
        self._scan_y = 14 if self._scan_y >= self.CANVAS_H - 14 else self._scan_y + (self.CANVAS_H - 28) / 26

        if frames_left > 0:
            self._scan_job = self.after(45, lambda: self._animate_scan(frames_left - 1))
        else:
            if self.current_image is not None:
                self._render_cover(self.current_image)
            self._draw_corners()
            # ---------------------------------------------------------------
            # PONTO DE INTEGRAÇÃO DO MODELO
            # Troque as duas linhas abaixo pela inferência real usando
            # self.current_image (objeto PIL.Image já carregado/capturado).
            simulated_age = random.randint(18, 58)
            simulated_confidence = random.randint(70, 95)
            # ---------------------------------------------------------------
            self.on_result((simulated_age, simulated_confidence))


# ---------------------------------------------------------------------------
# Painel de resultado
# ---------------------------------------------------------------------------
class ResultPanel(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color=PANEL, corner_radius=4,
                          border_width=1, border_color=LINE_SOFT)

        ctk.CTkLabel(
            self, text="RESULTADO", text_color=MUTED,
            font=font(FONT_MONO, 11), anchor="w",
        ).pack(fill="x", padx=20, pady=(18, 14))

        self.empty_label = ctk.CTkLabel(
            self, text="Aguardando imagem para estimar a idade.",
            text_color=MUTED_DIM, font=font(FONT_MONO, 11),
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
            age_row, text=" anos (estimado)", text_color=MUTED,
            font=font(FONT_BODY, 13),
        ).pack(side="left", padx=(6, 0), pady=(14, 0))

        self.age_range = ctk.CTkLabel(
            self.body, text="", text_color=MUTED,
            font=font(FONT_MONO, 11), anchor="w",
        )
        self.age_range.pack(fill="x", pady=(2, 18))

        conf_row = ctk.CTkFrame(self.body, fg_color="transparent")
        conf_row.pack(fill="x")
        ctk.CTkLabel(
            conf_row, text="CONFIANÇA", text_color=MUTED,
            font=font(FONT_MONO, 10),
        ).pack(side="left")
        self.confidence_value = ctk.CTkLabel(
            conf_row, text="0%", text_color=MUTED, font=font(FONT_MONO, 10),
        )
        self.confidence_value.pack(side="right")

        self.progress = ctk.CTkProgressBar(
            self.body, height=4, corner_radius=2,
            fg_color=LINE_SOFT, progress_color=CYAN,
        )
        self.progress.set(0)
        self.progress.pack(fill="x", pady=(6, 20))

        ctk.CTkLabel(
            self.body, text="●  Saída simulada — modelo ainda não integrado",
            text_color=MUTED_DIM, font=font(FONT_MONO, 10),
            fg_color=BG, corner_radius=14,
        ).pack(anchor="w", ipadx=10, ipady=5)

        self.reset_btn = ctk.CTkButton(
            self, text="Nova análise", fg_color="transparent",
            hover_color=PANEL_2, text_color=MUTED, font=font(FONT_MONO, 11),
            anchor="w", width=1, height=20,
        )

    def show_processing(self):
        self.body.pack_forget()
        self.reset_btn.pack_forget()
        self.empty_label.configure(text="Analisando imagem…")
        self.empty_label.pack(fill="x", padx=20)

    def show_result(self, age, confidence):
        self.empty_label.pack_forget()
        self.body.pack(fill="x", padx=20, pady=(0, 4))
        self.age_number.configure(text=str(age))
        self.age_range.configure(text=f"Faixa provável: {age-3}–{age+3} anos")
        self.confidence_value.configure(text=f"{confidence}%")
        self.progress.set(confidence / 100)
        self.reset_btn.pack(anchor="w", padx=20, pady=(0, 16))

    def reset(self):
        self.body.pack_forget()
        self.reset_btn.pack_forget()
        self.empty_label.configure(text="Aguardando imagem para estimar a idade.")
        self.empty_label.pack(fill="x", padx=20)
        self.progress.set(0)


# ---------------------------------------------------------------------------
# Tela 2 — Captura e resultado
# ---------------------------------------------------------------------------
class AppFrame(ctk.CTkFrame):
    def __init__(self, master, on_back):
        super().__init__(master, fg_color=BG)

        # cabeçalho ------------------------------------------------------
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(22, 18))

        ctk.CTkButton(
            header, text="‹  Início", fg_color="transparent",
            hover_color=PANEL, text_color=MUTED, font=font(FONT_BODY, 13),
            width=70, command=on_back,
        ).pack(side="left")

        if os.path.exists(LOGO_PATH):
            header_logo_img = Image.open(LOGO_PATH).convert("RGBA")
            self._header_logo = ctk.CTkImage(
                light_image=header_logo_img, dark_image=header_logo_img, size=(28, 28)
            )
            ctk.CTkLabel(header, image=self._header_logo, text="").pack(side="left", padx=(14, 6))

        ctk.CTkLabel(
            header, text="FRM", text_color=CREAM,
            font=font(FONT_DISPLAY, 20, weight="bold"),
        ).pack(side="left", expand=True)

        ctk.CTkLabel(
            header, text="●  Modelo em treinamento", text_color=BLUE,
            font=font(FONT_MONO, 11), fg_color=PANEL, corner_radius=14,
        ).pack(side="right", ipadx=10, ipady=5)

        ctk.CTkFrame(self, fg_color=LINE_SOFT, height=1).pack(fill="x", padx=28)

        # corpo ------------------------------------------------------------
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=28, pady=22)

        left = ctk.CTkFrame(body, fg_color="transparent")
        left.pack(side="left", padx=(0, 24))

        self.result_panel = ResultPanel(None)  # criado abaixo, referência temporária
        self.viewport = Viewport(left, on_result=self._handle_result)
        self.viewport.pack()
        self.webcam_error = ctk.CTkLabel(
            left, text="", text_color=MUTED, font=font(FONT_MONO, 11),
            wraplength=Viewport.CANVAS_W, justify="left",
        )
        self.webcam_error.pack(fill="x", pady=(10, 0))

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
        ctk.CTkButton(
            actions, text="Usar webcam", fg_color=BTN_BLUE, hover_color=BTN_BLUE_HOVER,
            border_width=0, text_color="white",
            font=font(FONT_BODY, 13, weight="bold"), height=44, corner_radius=6,
            command=self.use_webcam,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.result_panel = ResultPanel(right)
        self.result_panel.pack(fill="x")
        self.result_panel.reset_btn.configure(command=self.reset_all)

    # -- ações -------------------------------------------------------------
    def upload_photo(self):
        path = filedialog.askopenfilename(
            title="Selecionar foto",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        self.webcam_error.configure(text="")
        img = Image.open(path).convert("RGB")
        self.viewport.show_image(img)

    def use_webcam(self):
        self.webcam_error.configure(text="")
        try:
            self.viewport.start_webcam()
        except RuntimeError:
            self.webcam_error.configure(
                text="Não foi possível acessar a webcam. Verifique se ela está "
                     "conectada e se outra aplicação não está usando o dispositivo."
            )

    def reset_all(self):
        self.viewport.reset()
        self.result_panel.reset()
        self.webcam_error.configure(text="")

    def _handle_result(self, result):
        if result is None:
            self.result_panel.show_processing()
        else:
            age, confidence = result
            self.result_panel.show_result(age, confidence)

    def on_leave(self):
        self.viewport.stop_webcam()


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------
class FRMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FRM (Face Recognition Module) — Estimativa de idade")
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
        self.app_frame.on_leave()
        self.app_frame.lower()
        self.intro_frame.lift()


if __name__ == "__main__":
    app = FRMApp()
    app.mainloop()