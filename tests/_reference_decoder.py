"""A verbatim transcription of ComfyUI's `get_components_internal` frame loop.

THE ORACLE. `minimax_refpack.media` no longer calls `VideoFromFile`, so the only thing
that can say the streaming rewrite still produces the same pixels is the decoder it
replaced. This is that decoder, copied out of
CU/comfy_api/latest/_input_impl/video_types.py:308-448 with nothing changed but the
names it reaches for - it needs only av, numpy and torch, so it runs in this venv where
the real one cannot (comfy_api is not importable outside a ComfyUI process).

`test_the_transcription_still_matches_comfyui` re-checks it against the real class when
MINIMAX_REFPACK_COMFYUI names a checkout; without that it is a transcription asserted
against nothing but review, which is why the check exists at all.

Deliberately NOT kept honest by sharing code with media.py: an oracle that imports the
implementation proves the implementation equals itself.
"""

import itertools

import av
import numpy as np
import torch


def reference_decode(path, start_time=0.0, duration=0.0):
    """(images [N,H,W,3] float32, audio dict or None, frame_rate) - ComfyUI's own."""
    with av.open(path) as container:
        video_stream = next(s for s in container.streams if s.type == "video")

        frames = []
        audio_frames = []
        alphas = None
        start_pts = int(start_time / video_stream.time_base)
        end_pts = int((start_time + duration) / video_stream.time_base)

        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        image_format = "gbrpf32le"

        def process_image_format(a):
            return a

        align_graph = None
        audio = None

        streams = [video_stream]
        has_first_audio_frame = False
        checked_alpha = False

        video_done = False
        audio_done = True

        audio_stream = next(
            (s for s in reversed(container.streams.audio) if s.codec_context is not None),
            None,
        )
        if audio_stream is not None:
            streams += [audio_stream]
            resampler = av.audio.resampler.AudioResampler(format="fltp")
            audio_done = False

        for packet in container.demux(*streams):
            if video_done and audio_done:
                break

            if packet.stream.type == "video":
                if video_done:
                    continue
                try:
                    for frame in packet.decode():
                        if frame.pts < start_pts:
                            continue
                        if duration and frame.pts >= end_pts:
                            video_done = True
                            break

                        if not checked_alpha:
                            alpha_channel = False
                            for comp in frame.format.components:
                                if comp.is_alpha or frame.format.name == "pal8":
                                    alphas = []
                                    alpha_channel = True
                                    break
                            if frame.format.name in (
                                "yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8",
                            ):
                                def process_image_format(a):  # noqa: F811
                                    return a.float() / 255.0

                                image_format = "rgba" if alpha_channel else "rgb24"
                            else:
                                def process_image_format(a):  # noqa: F811
                                    return a

                                image_format = "gbrapf32le" if alpha_channel else "gbrpf32le"

                            checked_alpha = True

                        if image_format in ("gbrpf32le", "gbrapf32le") and frame.width % 32 != 0:
                            if align_graph is None:
                                pad_w = ((frame.width + 31) // 32) * 32
                                pad_h = ((frame.height + 31) // 32) * 32
                                g = av.filter.Graph()
                                g_src = g.add_buffer(
                                    width=frame.width, height=frame.height,
                                    format=frame.format.name, time_base=video_stream.time_base,
                                )
                                g_pad = g.add("pad", f"{pad_w}:{pad_h}:0:0")
                                g_fill = g.add(
                                    "fillborders",
                                    f"left=0:right={pad_w - frame.width}:top=0:"
                                    f"bottom={pad_h - frame.height}:mode=smear",
                                )
                                g_sink = g.add("buffersink")
                                g_src.link_to(g_pad)
                                g_pad.link_to(g_fill)
                                g_fill.link_to(g_sink)
                                g.configure()
                                align_graph = (g, g_src, g_sink)
                            align_graph[1].push(frame)
                            img = np.ascontiguousarray(
                                align_graph[2].pull().to_ndarray(format=image_format)[
                                    : frame.height, : frame.width
                                ]
                            )
                        else:
                            img = frame.to_ndarray(format=image_format)
                        if frame.rotation != 0:
                            k = int(round(frame.rotation // 90))
                            img = np.rot90(img, k=k, axes=(0, 1)).copy()
                        if alphas is None:
                            frames.append(torch.from_numpy(img))
                        else:
                            frames.append(torch.from_numpy(img[..., :-1]))
                            alphas.append(torch.from_numpy(img[..., -1:]))
                except av.error.InvalidDataError:
                    pass

            elif packet.stream.type == "audio":
                if audio_done:
                    continue

                aframes = itertools.chain.from_iterable(
                    map(resampler.resample, packet.decode())
                )
                for frame in aframes:
                    if duration and frame.time > start_time + duration:
                        audio_done = True
                        break

                    if not has_first_audio_frame:
                        offset_seconds = start_time - frame.pts * audio_stream.time_base
                        to_skip = max(0, int(offset_seconds * audio_stream.sample_rate))
                        if to_skip < frame.samples:
                            has_first_audio_frame = True
                            audio_frames.append(frame.to_ndarray()[..., to_skip:])
                    else:
                        audio_frames.append(frame.to_ndarray())

        images = (
            process_image_format(torch.stack(frames))
            if len(frames) > 0
            else torch.zeros(0, 0, 0, 3)
        )

        frame_rate = (
            __import__("fractions").Fraction(video_stream.average_rate)
            if video_stream.average_rate
            else __import__("fractions").Fraction(1)
        )

        if len(audio_frames) > 0:
            audio_data = np.concatenate(audio_frames, axis=1)
            if duration:
                audio_data = audio_data[..., : int(duration * audio_stream.sample_rate)]
            audio = {
                "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                "sample_rate": int(audio_stream.sample_rate) if audio_stream.sample_rate else 1,
            }

        return images, audio, frame_rate
