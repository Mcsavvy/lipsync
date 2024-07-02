from __future__ import annotations
import os
import customtkinter  # type: ignore[import]
from CTkMessagebox import CTkMessagebox  # type: ignore[import]
from startfile import startfile  # type: ignore[import]


from PIL import Image, ImageTk
import cv2

from invoke import (
    load_database,
    main,
    ModelQuality,
    ModelVersion,
    OutputHeight,
    Coordinates,
    FaceMask,
)
from .color import (  # type: ignore[import]
    PRIMARY_COLOR,
    PRIMARY_COLOR_2,
    SECONDARY_COLOR,
    NEUTRAL_COLOR,
    NEUTRAL_COLOR_1,
)
from customtkinter import (
    BooleanVar,
    IntVar,
    StringVar,
    DoubleVar,
    CTkCheckBox,
    CTkEntry,
    CTkLabel,
)


customtkinter.set_appearance_mode("light")


def NumberInput(var: IntVar, frame: customtkinter.CTkFrame) -> CTkEntry:
    return customtkinter.CTkEntry(
        frame,
        textvariable=var,
        width=50,
        height=20,
        border_width=1,
        border_color=NEUTRAL_COLOR,
        text_color=NEUTRAL_COLOR,
    )


def CheckBoxInput(
    var: BooleanVar, frame: customtkinter.CTkFrame, text: str
) -> CTkCheckBox:
    return customtkinter.CTkCheckBox(
        frame,
        variable=var,
        checkbox_height=20,
        checkbox_width=20,
        text=text,
        text_color=NEUTRAL_COLOR,
    )


def NumberInputLabel(frame: customtkinter.CTkFrame, text: str) -> CTkLabel:
    return customtkinter.CTkLabel(frame, text=text, text_color=NEUTRAL_COLOR)


def file_options_init(
    app: "App",
    frame: customtkinter.CTkFrame,
):
    preview_label: customtkinter.CTkLabel

    def open_video_image_dialog():
        file_types = (
            ("Video files", "*.mp4 *.avi"),
            ("Image files", "*.jpg *.jpeg *.png"),
        )

        file_obj = customtkinter.filedialog.askopenfile(
            title="Select files", filetypes=file_types
        )

        if file_obj:
            file_name_label.configure(text=os.path.basename(file_obj.name))
            app.face_var.set(file_obj.name)
            display_preview(file_obj.name)
        else:
            file_name_label.configure(text="No Video/Image")
            app.face_var.set("")
            display_preview("")

    def display_preview(file_path: str):
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(file_path)
            img.thumbnail((1000, 200))  # Resize for preview
            img_preview = ImageTk.PhotoImage(img)

            # Display image preview
            preview_label.configure(image=img_preview, text="")
            preview_label.image = img_preview  # Keep a reference
        elif file_path.lower().endswith((".mp4", ".avi")):
            # Display video preview (first frame)
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                preview_label.configure(text="")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail(
                    (1000, 200), Image.Resampling.LANCZOS
                )  # Resize for preview
                img_preview = ImageTk.PhotoImage(img)
                preview_label.configure(image=img_preview, text="")
                preview_label.image = img_preview  # Keep a reference
            cap.release()

    def open_audio_dialog():
        file_types = (("Audio files", "*.mp3 *.wav *.aac"),)

        file_obj = customtkinter.filedialog.askopenfile(
            title="Select Audio File", filetypes=file_types
        )

        if file_obj:
            file_audio_label.configure(text=os.path.basename(file_obj.name))
            app.audio_var.set(file_obj.name)
        else:
            file_audio_label.configure(text="No Audio")
            app.audio_var.set("")

    # BUG: placeholder not sowing when variable is passed
    # # title input
    # sync_title_input = customtkinter.CTkEntry(
    #     frame,
    #     width=app.width / 2,
    #     placeholder_text="name of this sync",
    #     placeholder_text_color="#808080",
    #     text_color=NEUTRAL_COLOR,
    #     textvariable=app.output_var,
    # )
    # sync_title_input._activate_placeholder()
    # sync_title_input.pack(anchor="w", padx=(20, 20), pady=(20, 0), ipady=10)

    # open image or vide

    # Preview label for image or video
    preview_label = customtkinter.CTkLabel(
        frame,
        text="Image/Video Preview",
        text_color=NEUTRAL_COLOR_1,
        bg_color="#808080",
        corner_radius=10,
        width=app.width / 2,
        height=200,
    )
    preview_label.pack(anchor="w", padx=(20, 20), pady=(20, 0))
    file_name_label = customtkinter.CTkLabel(
        frame,
        text="No Video/Image",
        fg_color=NEUTRAL_COLOR,
        text_color=NEUTRAL_COLOR_1,
        corner_radius=10,
        width=app.width / 2,
    )
    file_name_label.pack(
        anchor="w",
        padx=(20, 20),
        pady=(10, 0),
        ipady=10,
    )
    open_image_video = customtkinter.CTkButton(
        frame,
        text="Upload Image/Video",
        width=app.width / 2,
        fg_color=PRIMARY_COLOR,
        hover_color=PRIMARY_COLOR,
        corner_radius=10,
        command=open_video_image_dialog,
    )
    open_image_video.pack(anchor="w", padx=(20, 20), pady=(20, 0), ipady=10)

    # open audio
    file_audio_label = customtkinter.CTkLabel(
        frame,
        text="No Audio",
        fg_color=NEUTRAL_COLOR,
        text_color=NEUTRAL_COLOR_1,
        corner_radius=10,
        width=app.width / 2,
    )
    file_audio_label.pack(
        anchor="w",
        padx=(20, 20),
        pady=(50, 0),
        ipady=10,
    )

    open_audio = customtkinter.CTkButton(
        frame,
        text="Upload Audio",
        width=app.width / 2,
        fg_color=PRIMARY_COLOR,
        hover_color=PRIMARY_COLOR,
        corner_radius=10,
        command=open_audio_dialog,
    )
    open_audio.pack(anchor="w", padx=(20, 20), pady=(20, 0), ipady=10)


def crop_options_init(app: "App", frame: customtkinter.CTkFrame):
    croppad_label = customtkinter.CTkLabel(
        frame, text="Mouth Cropping", text_color=NEUTRAL_COLOR
    )
    croppad_label.grid(sticky="w", padx=(20, 2), row=0, column=0, columnspan=2)
    crop_top_label = NumberInputLabel(frame, text="top")
    crop_top_label.grid(sticky="w", padx=(20, 5), row=1, column=0)
    crop_top_entry = NumberInput(app.crop_top_var, frame)
    crop_top_entry.grid(sticky="w", padx=(0, 20), row=1, column=1)

    crop_bottom_label = NumberInputLabel(frame, text="bottom")
    crop_bottom_label.grid(sticky="w", padx=(20, 5), row=1, column=2)
    crop_bottom_entry = NumberInput(app.crop_bottom_var, frame)
    crop_bottom_entry.grid(sticky="w", padx=(0, 0), row=1, column=3)

    crop_left_label = NumberInputLabel(frame, text="left")
    crop_left_label.grid(sticky="w", padx=(20, 5), row=1, column=4)
    crop_left_entry = NumberInput(app.crop_left_var, frame)
    crop_left_entry.grid(sticky="w", padx=(0, 0), row=1, column=5)

    crop_right_label = NumberInputLabel(frame, text="right")
    crop_right_label.grid(sticky="w", padx=(20, 5), row=1, column=6)
    crop_right_entry = NumberInput(app.crop_right_var, frame)
    crop_right_entry.grid(sticky="w", padx=(0, 0), row=1, column=7)


def padding_options_init(app: "App", frame: customtkinter.CTkFrame):
    padding_label = customtkinter.CTkLabel(
        frame, text="Mouth Padding", text_color=NEUTRAL_COLOR
    )
    padding_label.grid(sticky="w", padx=(20, 5), row=0, column=0, columnspan=2)

    pad_top_label = NumberInputLabel(frame, text="top")
    pad_top_label.grid(sticky="w", padx=(20, 5), row=1, column=0)
    pad_top_entry = NumberInput(app.pad_top_var, frame)
    pad_top_entry.grid(sticky="w", padx=(0, 20), row=1, column=1)

    pad_bottom_label = NumberInputLabel(frame, text="bottom")
    pad_bottom_label.grid(sticky="w", padx=(20, 5), row=1, column=2)
    pad_bottom_entry = NumberInput(app.pad_bottom_var, frame)
    pad_bottom_entry.grid(sticky="w", padx=(0, 0), row=1, column=3)

    pad_left_label = NumberInputLabel(frame, text="left")
    pad_left_label.grid(sticky="w", padx=(20, 5), row=1, column=4)
    pad_left_entry = NumberInput(app.pad_left_var, frame)
    pad_left_entry.grid(sticky="w", padx=(0, 0), row=1, column=5)

    pad_right_label = NumberInputLabel(frame, text="right")
    pad_right_label.grid(sticky="w", padx=(20, 5), row=1, column=6)
    pad_right_entry = NumberInput(app.pad_right_var, frame)
    pad_right_entry.grid(sticky="w", padx=(0, 0), row=1, column=7)


def bounding_box_options_init(app: "App", frame: customtkinter.CTkFrame):
    bounding_label = customtkinter.CTkLabel(
        frame, text="Bounding Box", text_color=NEUTRAL_COLOR
    )
    bounding_label.grid(sticky="w", padx=(20, 5), row=0, column=0, columnspan=2)

    bounding_top_label = NumberInputLabel(frame, text="top")
    bounding_top_label.grid(sticky="w", padx=(20, 5), row=1, column=0)
    bounding_top_entry = NumberInput(app.bounding_top_var, frame)
    bounding_top_entry.grid(sticky="w", padx=(0, 20), row=1, column=1)

    bounding_bottom_label = NumberInputLabel(frame, text="bottom")
    bounding_bottom_label.grid(sticky="w", padx=(20, 5), row=1, column=2)
    bounding_bottom_entry = NumberInput(app.bounding_bottom_var, frame)
    bounding_bottom_entry.grid(sticky="w", padx=(0, 0), row=1, column=3)

    bounding_left_label = NumberInputLabel(frame, text="left")
    bounding_left_label.grid(sticky="w", padx=(20, 5), row=1, column=4)
    bounding_left_entry = NumberInput(app.bounding_left_var, frame)
    bounding_left_entry.grid(sticky="w", padx=(0, 0), row=1, column=5)

    bounding_right_label = NumberInputLabel(frame, text="right")
    bounding_right_label.grid(sticky="w", padx=(20, 5), row=1, column=6)
    bounding_right_entry = NumberInput(app.bounding_right_var, frame)
    bounding_right_entry.grid(sticky="w", padx=(0, 0), row=1, column=7)


def rendering_options_init(app: "App", frame: customtkinter.CTkFrame):
    rendering_label = customtkinter.CTkLabel(
        frame, text="Rendering Options", text_color=NEUTRAL_COLOR
    )
    rendering_label.grid(sticky="w", padx=(20, 5), row=0, column=0, columnspan=2)

    static = CheckBoxInput(app.static_var, frame, "Static")
    static.grid(sticky="w", padx=(20, 5), row=1, column=0)

    rotate = CheckBoxInput(app.rotate_var, frame, "Rotate")
    rotate.grid(sticky="w", padx=(20, 5), row=1, column=1)

    smooth = CheckBoxInput(app.smooth_var, frame, "Smooth")
    smooth.grid(sticky="w", padx=(20, 5), row=1, column=2)

    super_resolution = CheckBoxInput(
        app.super_resolution_var, frame, "Super Resolution"
    )
    super_resolution.grid(sticky="w", padx=(20, 5), row=1, column=3)

    height_label = customtkinter.CTkLabel(
        frame, text="Output Height", text_color=NEUTRAL_COLOR
    )
    height_label.grid(sticky="w", padx=(20, 5), pady=(15, 0), row=2, column=0)
    height = customtkinter.CTkComboBox(
        frame,
        values=["full resolution", "half resolution"],
        variable=app.height_var,
    )
    height.grid(sticky="w", padx=(20, 5), pady=(15, 0), row=2, column=1)

    fps_label = customtkinter.CTkLabel(
        frame, text="Frames Per Second", text_color=NEUTRAL_COLOR
    )
    fps_label.grid(sticky="w", padx=(20, 5), pady=(15, 0), row=2, column=2)
    fps = NumberInput(app.fps_var, frame)
    fps.grid(sticky="w", padx=(20, 5), pady=(15, 0), row=2, column=3)


def mask_options_init(app: "App", frame: customtkinter.CTkFrame):
    mask_label = customtkinter.CTkLabel(
        frame, text="Mask Options", text_color=NEUTRAL_COLOR
    )
    mask_label.grid(sticky="w", padx=(20, 5), row=0, column=0, columnspan=2)

    mask_size_label = customtkinter.CTkLabel(
        frame, text="Size", text_color=NEUTRAL_COLOR
    )
    mask_size_label.grid(sticky="w", padx=(20, 5), row=1, column=0)
    mask_size = NumberInput(app.mask_size_var, frame)
    mask_size.grid(sticky="w", padx=(0, 0), row=1, column=1)

    mask_feathering_label = customtkinter.CTkLabel(
        frame, text="Feathering", text_color=NEUTRAL_COLOR
    )
    mask_feathering_label.grid(sticky="w", padx=(20, 5), row=1, column=2)
    mask_feathering = NumberInput(app.mask_feathering_var, frame)
    mask_feathering.grid(sticky="w", padx=(0, 0), row=1, column=3)

    mask_mouth_tracking = CheckBoxInput(
        app.mask_mouth_tracking, frame, "Mouth Tracking"
    )
    mask_mouth_tracking.grid(sticky="w", padx=(40, 5), row=1, column=4)

    debug_mask = CheckBoxInput(app.debug_mask, frame, "Debug Mask")
    debug_mask.grid(sticky="w", padx=(20, 5), row=1, column=5)


def model_options_init(app: "App", frame: customtkinter.CTkFrame):
    model_label = customtkinter.CTkLabel(
        frame, text="Model Options", text_color=NEUTRAL_COLOR
    )
    model_label.grid(sticky="w", padx=(20, 5), row=0, column=0, columnspan=2)

    model_select_label = customtkinter.CTkLabel(
        frame, text="Version", text_color=NEUTRAL_COLOR
    )
    model_select_label.grid(sticky="w", padx=(20, 5), row=1, column=0)
    model_select = customtkinter.CTkComboBox(
        frame, values=["wav2lip", "wav2lip_GAN"], variable=app.version_var, width=120
    )
    model_select.grid(sticky="w", padx=(0, 0), row=1, column=1)

    model_quality_label = customtkinter.CTkLabel(
        frame, text="Quality", text_color=NEUTRAL_COLOR
    )
    model_quality_label.grid(sticky="w", padx=(20, 5), row=1, column=2)
    model_quality_select = customtkinter.CTkComboBox(
        frame,
        values=["Improved", "Fast", "Enhanced"],
        variable=app.quality_var,
        width=100,
    )
    model_quality_select.grid(sticky="w", padx=(0, 0), row=1, column=3)

    upscaler_label = customtkinter.CTkLabel(
        frame, text="Upscaler", text_color=NEUTRAL_COLOR
    )
    upscaler_label.grid(sticky="w", padx=(20, 5), row=1, column=4)
    upscaler = customtkinter.CTkComboBox(
        frame, values=["gfpgan", "RestoreFormer"], variable=app.upscaler_var, width=130
    )
    upscaler.grid(sticky="w", padx=(0, 0), row=1, column=5)


def history_init(app: "App", frame: customtkinter.CTkFrame):
    def refresh_history():
        # Code to refresh the history goes here
        history_label_container.destroy()
        refresh_button.destroy()
        for widget in frame.winfo_children():
            widget.destroy()
        history_init(app, frame)

    history_label_container = customtkinter.CTkFrame(
        frame,
        width=app.width / 2,
        height=40,
        fg_color=SECONDARY_COLOR,
    )
    history_label_container.pack(side="top", fill="x", expand=False, pady=0, padx=0)
    history_label_container.grid_propagate(flag=False)

    history_label = customtkinter.CTkLabel(
        history_label_container,
        text="History",
        text_color=NEUTRAL_COLOR,
        font=("Open Sans", 14, "bold"),
    )
    history_label.grid(row=0, column=0, pady=(0, 0), padx=(20, 0), columnspan=2)

    refresh_button = customtkinter.CTkButton(
        history_label_container,
        text="↺",
        width=7,
        height=2,
        fg_color=SECONDARY_COLOR,
        hover_color=PRIMARY_COLOR,
        corner_radius=20,
        command=refresh_history,
        font=("Open Sans", 14, "bold"),
    )
    refresh_button.grid(row=0, column=2, pady=(0, 0), padx=(2, 20), sticky="e")

    def get_video_thumbnail(video_path: str) -> ImageTk.PhotoImage:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            # create a small thumbnail
            img.thumbnail((40, 40), Image.Resampling.LANCZOS)  # Resize for preview
            img_preview = ImageTk.PhotoImage(img)
        else:
            raise ValueError("Cannot read video file")
        cap.release()
        return img_preview

    database = load_database()
    for i, run_id in enumerate(database):
        timestamp = database[run_id]["timestamp"]
        data = database[run_id]["params"]
        try:
            image = get_video_thumbnail(data["output"])
        except Exception as e:
            print("Error: ", e)
            continue
        title = data.get("output", "No Title")

        history_frame = customtkinter.CTkFrame(
            frame,
            width=app.width / 2,
            height=60,
            fg_color="#808080",
        )
        history_frame.pack(side="top", anchor="w", pady=(0, 20), padx=0)
        history_frame.grid_propagate(flag=False)

        customtkinter.CTkLabel(
            history_frame,
            image=image,
            text="",
            text_color=NEUTRAL_COLOR,
            bg_color=NEUTRAL_COLOR,
            corner_radius=0,
            width=10,
            height=10,
        ).grid(row=0, column=0, pady=5, padx=5, rowspan=2)
        customtkinter.CTkLabel(
            history_frame,
            text=title,
            text_color=NEUTRAL_COLOR,
            bg_color="#808080",
            corner_radius=10,
        ).grid(row=0, column=1, columnspan=2, sticky="w")
        customtkinter.CTkLabel(
            history_frame,
            text=timestamp,
            text_color="white",
            bg_color="#808080",
            corner_radius=10,
            font=("Open Sans", 10, "italic"),
        ).grid(row=1, column=1, sticky="w")
        customtkinter.CTkButton(
            history_frame,
            text="Open",
            width=7,
            height=2,
            fg_color=PRIMARY_COLOR,
            hover_color=PRIMARY_COLOR,
            corner_radius=10,
            command=lambda: startfile(data["output"]),
        ).grid(row=1, column=2, sticky="e")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # file variables
        self.face_var = StringVar(self)
        self.audio_var = StringVar(self)
        self.output_var = StringVar(self)

        # model variables
        self.quality_var = StringVar(self, value="Fast")
        self.version_var = StringVar(self, value="Wav2Lip")
        self.upscaler_var = StringVar(self, value="gfpgan")

        # rendering variables
        self.fps_var = IntVar(self, value=25)
        self.static_var = BooleanVar(self, value=False)
        self.rotate_var = BooleanVar(self, value=False)
        self.smooth_var = BooleanVar(self, value=False)
        self.height_var = StringVar(self, value="full resolution")
        self.super_resolution_var = BooleanVar(self, value=True)

        # padding variables
        self.pad_top_var = IntVar(self, value=0)
        self.pad_bottom_var = IntVar(self, value=0)
        self.pad_left_var = IntVar(self, value=0)
        self.pad_right_var = IntVar(self, value=0)

        # mask variables
        self.mask_size_var = DoubleVar(self, value=2.5)
        self.mask_feathering_var = IntVar(self, value=2)
        self.mask_mouth_tracking = BooleanVar(self, value=False)
        self.debug_mask = BooleanVar(self, value=False)

        # crop variables
        self.crop_top_var = IntVar(self, value=0)
        self.crop_bottom_var = IntVar(self, value=-1)
        self.crop_left_var = IntVar(self, value=0)
        self.crop_right_var = IntVar(self, value=-1)

        # bounding box variables
        self.bounding_top_var = IntVar(self, value=-1)
        self.bounding_bottom_var = IntVar(self, value=-1)
        self.bounding_left_var = IntVar(self, value=-1)
        self.bounding_right_var = IntVar(self, value=-1)

        # generating size based on user screen
        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.custom_size = str(self.width) + "x" + str(self.height)

        # fonts
        self.font_14 = ("Open Sans", 14, "bold")
        self.font_18 = ("Open Sans", 18, "bold")

        self.title("Lip Sync")
        self.geometry(self.custom_size)

        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.pack(side="top", fill="both", expand=True)
        self.tabview.add("Home")
        self.tabview.add("Sync")

        self.home_tab = self.tabview.tab("Home")
        self.sync_tab = self.tabview.tab("Sync")

        self.init_home_tab()
        self.init_sync_tab()

    def init_home_tab(self):
        header_frame = customtkinter.CTkFrame(
            self.home_tab,
            width=self.width,
            bg_color=PRIMARY_COLOR,
            fg_color=PRIMARY_COLOR,
        )
        header_frame.pack(side="top", fill="x", expand=False, pady=0, padx=0)

        header_text_logo = customtkinter.CTkLabel(
            header_frame,
            text="LIP-SYNC",
            width=10,
            height=10,
            font=self.font_18,
            text_color=NEUTRAL_COLOR_1,
            bg_color=PRIMARY_COLOR,
        )
        header_text_logo.pack(side="left", padx=20, ipadx=5, pady=20, ipady=10)

        # Now, you can add your content to self.scrollable_frame instead of self.content_frame
        self.content_frame = customtkinter.CTkScrollableFrame(
            self.home_tab,
            width=self.width,
            height=self.height,
            bg_color=SECONDARY_COLOR,
            fg_color=SECONDARY_COLOR,
        )
        self.content_frame.pack(side="bottom")
        history_init(self, self.content_frame)

        self.tabview.set("Home")

    def init_sync_tab(self):
        self.setting = customtkinter.CTkScrollableFrame(
            self.sync_tab,
            width=self.width / 2,
            height=self.height,
            fg_color=SECONDARY_COLOR,
        )
        self.setting.pack(side="left", fill="both", expand=True)

        # model frame
        self.model_frame = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=80, fg_color=PRIMARY_COLOR
        )
        self.model_frame.pack(side="top", pady=(20, 0), padx=20)
        self.model_frame.grid_propagate(flag=False)
        model_options_init(self, self.model_frame)
        # static frame
        self.rendering_frame = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=100, fg_color=PRIMARY_COLOR
        )
        self.rendering_frame.pack(side="top", pady=(20, 0), padx=20)
        self.rendering_frame.grid_propagate(flag=False)
        rendering_options_init(self, self.rendering_frame)

        # padding
        self.padding = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=60, fg_color=PRIMARY_COLOR
        )
        self.padding.pack(side="top", pady=(20, 0), padx=20)
        self.padding.grid_propagate(flag=False)
        padding_options_init(self, self.padding)

        # cropping
        self.cropping = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=60, fg_color=PRIMARY_COLOR
        )
        self.cropping.pack(side="top", pady=(20, 0), padx=20)
        self.cropping.grid_propagate(flag=False)
        crop_options_init(self, self.cropping)

        # boundingbox
        self.boundingbox = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=60, fg_color=PRIMARY_COLOR
        )
        self.boundingbox.pack(side="top", pady=(20, 0), padx=20)
        self.boundingbox.grid_propagate(flag=False)
        bounding_box_options_init(self, self.boundingbox)

        # facemask
        self.mask = customtkinter.CTkFrame(
            self.setting, width=self.width / 2, height=60, fg_color=PRIMARY_COLOR
        )
        self.mask.pack(side="top", pady=(20, 0), padx=20)
        self.mask.grid_propagate(flag=False)
        mask_options_init(self, self.mask)

        # upload frame
        self.upload = customtkinter.CTkFrame(
            self.sync_tab,
            width=self.width / 2,
            height=self.height,
            fg_color=SECONDARY_COLOR,
        )
        self.upload.pack(side="right", expand=True, padx=(5, 0))
        self.upload.pack_propagate(flag=False)
        file_options_init(self, self.upload)

        self.submit_btn = customtkinter.CTkButton(
            self.upload,
            text="Sync Video",
            width=self.width / 2,
            fg_color=PRIMARY_COLOR,
            hover_color=PRIMARY_COLOR,
            corner_radius=10,
            command=self.sync,
        )
        self.submit_btn.pack(anchor="w", padx=(20, 20), pady=(70, 0), ipady=10)
        self.progress = customtkinter.CTkProgressBar(
            self.upload,
            width=self.width / 2,
            height=5,
            progress_color=PRIMARY_COLOR_2,
        )
        self.progress.set(0)
        self.progress.pack(anchor="w", padx=(20, 20), pady=(20, 0), ipady=10)

    def sync(self) -> None:
        face = self.face_var.get().strip()
        audio = self.audio_var.get().strip()
        output = self.output_var.get().strip() or None
        quality: ModelQuality = self.quality_var.get()
        version: ModelVersion = self.version_var.get()
        height: OutputHeight = self.height_var.get()
        upscaler = self.upscaler_var.get()
        smooth = self.smooth_var.get()
        padding: Coordinates = {
            "top": self.pad_top_var.get(),
            "bottom": self.pad_bottom_var.get(),
            "left": self.pad_left_var.get(),
            "right": self.pad_right_var.get(),
        }
        mask: FaceMask = {
            "size": self.mask_size_var.get(),
            "feathering": self.mask_feathering_var.get(),
            "mouth_tracking": self.mask_mouth_tracking.get(),
            "debug_mask": self.debug_mask.get(),
        }
        crop: Coordinates = {
            "top": self.crop_top_var.get(),
            "bottom": self.crop_bottom_var.get(),
            "left": self.crop_left_var.get(),
            "right": self.crop_right_var.get(),
        }
        bounding_box: Coordinates = {
            "top": self.bounding_top_var.get(),
            "bottom": self.bounding_bottom_var.get(),
            "left": self.bounding_left_var.get(),
            "right": self.bounding_right_var.get(),
        }
        rotate = self.rotate_var.get()
        static = self.static_var.get()
        fps = self.fps_var.get()

        if not (face and audio):
            CTkMessagebox(
                self.sync_tab,
                title="Cannot Proceed",
                icon="cancel",
                message="Please upload both video/image and audio file",
            )
            return
        self.submit_btn.configure(
            state="disabled", text="Syncing...", require_redraw=True
        )
        print("Syncing...")
        run = main(
            face=face,
            audio=audio,
            output=output,
            quality=quality,
            version=version,
            height=height,
            smooth=smooth,
            padding=padding,
            mask=mask,
            bounding_box=bounding_box,
            face_crop=crop,
            rotate=rotate,
            upscaler=upscaler,
            static=static,
            frames_per_second=fps,
        )
        try:
            data = next(run)
        except Exception as e:
            CTkMessagebox(
                self.sync_tab,
                title="Sync Error",
                icon="cancel",
                message=str(e),
            )
            self.submit_btn.configure(
                state="normal", text="Sync Video", require_redraw=True
            )
            return
        total: int = data["total"]
        output_file = data["output"]
        print("Total: ", total)
        step = 1 / total
        print("Step: ", step)
        self.progress.configure(determinate_speed=step)
        self.progress.set(0)
        self.progress.update()
        try:
            for i in run:
                self.progress.set(i["progress"] * step)
                self.progress.update()
        except Exception as e:
            CTkMessagebox(
                self.sync_tab,
                title="Sync Error",
                icon="cancel",
                message=str(e),
            )
            self.submit_btn.configure(
                state="normal", text="Sync Video", require_redraw=True
            )
            print("Error: ", e)
        else:
            msg = CTkMessagebox(
                self.sync_tab,
                title="Sync Complete",
                icon="info",
                message="Syncing complete",
                option_2="OPEN",
            )
            if msg.get() == "OPEN":
                startfile(output_file)

            print("Synced")
        self.progress.stop()
        self.submit_btn.configure(
            state="normal", text="Sync Video", require_redraw=True
        )
