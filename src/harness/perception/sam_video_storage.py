"""Retain SAM3 video inputs on CPU while its per-frame network runs on CUDA."""

import functools


class CPUFrameSequence:
    """Avoid SAM3's recursive dataclass transfer copying the entire video.

    SAM3 indexes this sequence before moving each selected frame to its model
    device. Deliberately do not expose ``to`` or subclass list: either would
    cause copy_data_to_device to transfer every frame during initialization.
    """

    def __init__(self, images):
        if images.device.type != "cpu":
            raise ValueError("CPU video storage requires a CPU tensor")
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if getattr(getattr(index, "device", None), "type", "cpu") != "cpu":
            index = index.cpu()
        return self.images[index]


def install_cpu_video_storage(predictor):
    """Adapt only the predictor instance; never edit the installed SAM3 source."""
    construct = predictor.model._construct_initial_input_batch

    @functools.wraps(construct)
    def construct_with_storage(state, images):
        cpu = images.device.type == "cpu"
        frames = CPUFrameSequence(images) if cpu else images
        construct(state, frames)
        retained = state["input_batch"].img_batch
        if cpu and retained is not frames:
            raise RuntimeError("SAM3 did not retain the requested CPU video storage")
        state["plumb_video_storage"] = {
            "device": "cpu" if cpu else str(images.device),
            "frame_count": len(images),
            "shape": list(images.shape),
            "dtype": str(images.dtype),
            "bytes": images.numel() * images.element_size(),
        }

    predictor.model._construct_initial_input_batch = construct_with_storage
    return predictor


def offload_video_to_cpu(request):
    value = request.get("offload_video_to_cpu", True)
    if type(value) is not bool:
        raise ValueError("offload_video_to_cpu must be a boolean")
    return value
