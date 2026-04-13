# preprocessing script for WLASL dataset
# 1. Convert .swf, .mkv file to mp4.
# 2. Extract YouTube frames and create video instances.

import os
import json
import cv2
import shutil


def convert_everything_to_mp4():
    import subprocess
    from pathlib import Path

    ffmpeg_exe = r"C:\Users\Micah Joyce\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

    src_path = Path("raw_videos")
    dst_path = Path("raw_videos_mp4")
    dst_path.mkdir(exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"raw_videos folder not found: {src_path.resolve()}")

    files = list(src_path.glob("*"))
    total = len(files)

    print(f"Found {total} file(s) in raw_videos")

    for i, src_file in enumerate(files, start=1):
        if not src_file.is_file():
            continue

        dst_file = dst_path / f"{src_file.stem}.mp4"

        if dst_file.exists():
            print(f"{i}/{total}: already exists -> {dst_file.name}")
            continue

        print(f"{i}/{total}: processing {src_file.name}")

        if src_file.suffix.lower() == ".mp4":
            shutil.copy2(src_file, dst_file)
        else:
            subprocess.run(
                [
                    ffmpeg_exe,
                    "-loglevel",
                    "panic",
                    "-i",
                    str(src_file),
                    "-vf",
                    "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                    str(dst_file),
                ],
                check=True,
            )


def video_to_frames(video_path, size=None):
    """
    video_path -> str, path to video.
    size -> (int, int), width, height.
    """

    cap = cv2.VideoCapture(video_path)

    frames = []
    
    while True:
        ret, frame = cap.read()
    
        if ret:
            if size:
                frame = cv2.resize(frame, size)
            frames.append(frame)
        else:
            break

    cap.release()

    return frames


def convert_frames_to_video(frame_array, path_out, size, fps=25):
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


def extract_frame_as_video(src_video_path, start_frame, end_frame):
    frames = video_to_frames(src_video_path)
    return frames[start_frame: end_frame+1]


def extract_all_yt_instances(content):
    cnt = 1

    if not os.path.exists('videos'):
        os.mkdir('videos')

    for entry in content:
        instances = entry['instances']

        for inst in instances:
            url = inst['url']
            video_id = inst['video_id']

            if 'youtube' in url or 'youtu.be' in url:
                cnt += 1
                
                yt_identifier = url[-11:]

                src_video_path = os.path.join('raw_videos_mp4', yt_identifier + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if not os.path.exists(src_video_path):
                    continue

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                start_frame = inst['frame_start'] - 1
                end_frame = inst['frame_end'] - 1

                if end_frame <= 0:
                    shutil.copyfile(src_video_path, dst_video_path)
                    continue

                selected_frames = extract_frame_as_video(src_video_path, start_frame, end_frame)

                if len(selected_frames) == 0:
                    continue
                
                size = selected_frames[0].shape[:2][::-1]
                convert_frames_to_video(selected_frames, dst_video_path, size)

                print(cnt, dst_video_path)
            else:
                cnt += 1

                src_video_path = os.path.join('raw_videos_mp4', video_id + '.mp4')
                dst_video_path = os.path.join('videos', video_id + '.mp4')

                if os.path.exists(dst_video_path):
                    print('{} exists.'.format(dst_video_path))
                    continue

                if not os.path.exists(src_video_path):
                    continue

                print(cnt, dst_video_path)
                shutil.copyfile(src_video_path, dst_video_path)

        
def main():
    convert_everything_to_mp4()
    content = json.load(open('WLASL_v0.3.json'))
    extract_all_yt_instances(content)


if __name__ == "__main__":
    main()