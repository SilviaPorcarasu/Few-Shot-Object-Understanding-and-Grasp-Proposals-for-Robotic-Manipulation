import json
from pathlib import Path

import pyrealsense2 as rs


def main():
    output_path = Path("data/intrinsics.json")
    output_path.parent.mkdir(exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)

    try:
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        data = {
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "model": str(color_intrinsics.model),
            "coeffs": list(color_intrinsics.coeffs),
        }

        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved intrinsics to {output_path}")
        print(json.dumps(data, indent=2))

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()