import cv2
from abc import ABC, abstractmethod
from src.video import get_video_info, prettify_video_info
from os import listdir
from os.path import isfile, join


class FrameGenerator:
    def __init__(
            self, source, frame_count, resolution, every_nth_frame=1, use_rgb=True
    ):
        self._frame_count = frame_count
        self.use_rgb = use_rgb
        self.every_nth_frame = every_nth_frame
        self.resolution = resolution
        self.source = source

    @abstractmethod
    def __iter__(self):
        pass

    def __len__(self):
        return self._frame_count // self.every_nth_frame


class FrameGeneratorVideo(FrameGenerator):
    def __init__(
            self, video_source, show_video_info=True, every_nth_frame=1, use_rgb=True
    ):
        """
        Init
        Parameters
        ----------
        video_source: str
            path to the video file you want to read.

        show_video_info: bool
            if true then video info is printed to the console

        every_nth_frame: int
            every nth frame will be yielded by the generator. 
            1 means that every single frame will be yielded.

        use_rgb: bool, Optional
            if True RGB image will be returned else the colore mode is cv2 default bgr.
        """

        video_file, frame_count, fps, length, height, width = get_video_info(
            video_source
        )
        if show_video_info:
            print(
                prettify_video_info(
                    video_source, frame_count, fps, length, height, width
                )
            )
        self.fps = fps
        self.length = length
        self._cap = cv2.VideoCapture(video_file)
        if not self._cap.isOpened():
            raise ValueError("could not open video file: {0}".format(video_file))
        super().__init__(source=video_source,
                         frame_count=frame_count,
                         resolution=(width, height),
                         every_nth_frame=every_nth_frame,
                         use_rgb=use_rgb)

    def __iter__(self):
        """ Read frame from an opencv capture objects and yields
        until this object is not closed.

        Returns
        -------
        a nupy array representing an RGB frame
        """
        cnt = -1
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            cnt += 1
            if cnt % self.every_nth_frame != 0:
                continue
            if frame is None:
                return
            if self.use_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        self._cap.release()
        raise StopIteration()


class FrameGeneratorImageSequence(FrameGenerator):
    def __init__(
            self, video_source, every_nth_frame=1, use_rgb=True
    ):

        self._video_files = [join(video_source, f) for f in listdir(video_source) if isfile(join(video_source, f))]
        self._video_files.sort()
        sample_img = cv2.imread(self._video_files[0])

        super().__init__(source=video_source,
                         frame_count=len(self._video_files),
                         resolution=sample_img.shape[1::-1],
                         every_nth_frame=every_nth_frame,
                         use_rgb=use_rgb)

    def __iter__(self):
        """ Read frame from an opencv capture objects and yields
        until this object is not closed.

        Returns
        -------
        a nupy array representing an RGB frame
        """
        for img_file in self._video_files[0::self.every_nth_frame]:
            frame = cv2.imread(img_file)
            if self.use_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame
        raise StopIteration()
