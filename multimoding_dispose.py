import asyncio
import shutil
import base64

from dashscope import MultiModalConversation
import api_key

import cv2
import os
from base import video_client
from langchain_core.output_parsers import JsonOutputParser



async def audio_text(path:str):
    print('正在进行语音识别')
    messages = [
        {
            "role": "user",
            "content": [
                {"audio": path},
                {"text": "直接输出这段语音说到的内容,不要加上这段语音的内容，直接输出用户说的话"}
            ]
        }
    ]
    response = await asyncio.to_thread(
        MultiModalConversation.call,
        model='qwen-audio-turbo-latest',
        messages=messages,
        api_key=api_key.qwen_api
    )

    if response.status_code == 200:
        transcribed_text = response.output.choices[0].message.content[0]['text']
        print("语音识别成功！")
        return transcribed_text
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误代码: {response.code}")
        print(f"错误信息: {response.message}")
        return None

async def audio_emotion(path:str):
    messages = [
        {
            "role": "user",
            "content": [
                {"audio": path},
                {"text": "这是面试者进行面试时面对面试官询问的问题进行的回答，我现在需要你进行情感分析，来判断面试者的抗压能力如何，你要从 语言连贯程度，紧张程度，是否结巴，是否存在语速过快等角度进行分析 返回结果以json格式输出，存在一个键值 'score' 对应类型为int类型 代表得到的评分 满分100分"}
            ]
        }
    ]
    response = await asyncio.to_thread(
        MultiModalConversation.call,
        model='qwen-audio-turbo-latest',
        messages=messages,
        api_key=api_key.qwen_api
    )

    if response.status_code == 200:
        transcribed_text = response.output.choices[0].message.content[0]['text']
        prase=JsonOutputParser()
        transcribed_text=prase.parse(transcribed_text)
        print("语音语调分析识别成功！")
        print(transcribed_text)
        return transcribed_text['score']
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误代码: {response.code}")
        print(f"错误信息: {response.message}")
        return 75

def extract_frames(interval_sec=1):
    """
    从视频中提取帧并保存为图片。
    :param video_path: 输入视频文件的路径。
    :param output_folder: 保存提取出的图片的文件夹路径。
    :param interval_sec: 提取帧的时间间隔（秒）。例如，1表示每秒提取一帧。
    """
    input_folder='面试视频（用户）'
    output_folder = 'video_picture'
    # 1. 无论是否存在，都先清空并重建主输出文件夹
    path = os.listdir(input_folder)
    paths= sorted(
        path,
        key=lambda x: os.path.getctime(os.path.join(input_folder, x)),
    )
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"已清空主文件夹: {output_folder}")
    os.makedirs(output_folder)
    print(f"已创建主文件夹: {output_folder}")

    # 检查输入视频文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹不存在于 {input_folder}")
        return



    # 筛选出视频文件
    video_files = [f for f in paths if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]
    if not video_files:
        print(f"在文件夹 {input_folder} 中未找到支持的视频文件。")
        return

    print(f"找到 {len(video_files)} 个视频文件，准备处理...")

    # 2. 遍历所有找到的视频文件
    for i, video_filename in enumerate(video_files, 1):
        # 关键修复：将输入文件夹路径和视频文件名拼接成完整路径
        full_video_path = os.path.join(input_folder, video_filename)
        print(f"\n--- 开始处理第 {i} 个视频: {video_filename} ---")

        # 为当前视频创建带序号的子文件夹
        video_subfolder_name = f'video_{i}_picture'
        video_specific_folder = os.path.join(output_folder, video_subfolder_name)
        os.makedirs(video_specific_folder)
        print(f"已创建子文件夹: {video_specific_folder}")

        # 使用完整的路径打开视频
        cap = cv2.VideoCapture(full_video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {full_video_path}")
            continue  # 跳过这个文件，继续处理下一个

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"视频帧率: {fps:.2f} FPS")

        frame_interval = int(fps * interval_sec) or 1
        print(f"每隔 {frame_interval} 帧提取一张图片 (大约每 {interval_sec} 秒一张)")

        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                output_path = os.path.join(video_specific_folder, f"frame_{saved_frame_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved_frame_count += 1
            frame_count += 1

        cap.release()
        print(f"视频 '{video_filename}' 处理完成！")
        print(f"总共处理了 {frame_count} 帧，在文件夹 '{video_specific_folder}' 中保存了 {saved_frame_count} 张图片。")

async def video_dispose(filename: str):
    """
    处理文件夹中的图片帧，通过API进行分析。
    """
    print('正在执行')
    try:

            filenames=os.listdir(filename)
            # 2. 【重要】对文件名进行排序，确保帧的顺序是正确的
            filenames.sort()
            # 3. 构建包含完整路径的列表
            full_path_list = [os.path.join(filename, f) for f in filenames if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not full_path_list:
                print(f"错误：在文件夹{filename}中没有找到图片文件。")
                return
            print(f"找到了 {len(full_path_list)} 张图片，准备进行分析...")
            # 4. 将每张图片读取并编码为Base64
            base64_images = []
            for image_path in full_path_list:
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    # API需要知道数据类型，所以我们使用Data URL格式
                    # 假设你的图片是jpg格式
                    base64_images.append(f"data:image/jpeg;base64,{encoded_string}")

            # 5. 调用API，传入Base64编码后的图片列表
            completion = await video_client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            # 这里传入的是Base64编码后的图片数据列表
                            "video": base64_images
                        },
                        {
                            "type": "text",
                            "text": "这是一个面试者，你需要对他进行面部微表情以及肢体语言的分析，返回结果以json 格式输出  存在一个键值 'score' 对应类型为int类型 代表得到的评分 满分100分"
                        }
                    ]
                }]
            )
            prase = JsonOutputParser()
            transcribed_text = prase.parse(completion.choices[0].message.content)
            print(transcribed_text)
            return transcribed_text['score']

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return 75
async def body_movement_analyse_full_flow():
    extract_frames()
    image_folder_path = 'video_picture'
    task_list=[]
    # 1. 获取文件夹中所有的文件名
    filenames = os.listdir(image_folder_path)
    for filename in filenames:
        filename = os.path.join(image_folder_path, filename)
        print(filename)
        task=asyncio.create_task(video_dispose(filename))
        task_list.append(task)
    ans=await asyncio.gather(*task_list)
    print(len(ans))
    return ans
async def emotion_analyse_full_flow():
    audio_folder_path ='语音资料（用户）'
    task_list = []
    paths = os.listdir(audio_folder_path)
    path = sorted(
        paths,
        key=lambda x: os.path.getctime(os.path.join(audio_folder_path, x)),
    )
    for filename in path:
        filename=os.path.join(audio_folder_path, filename)
        print(filename)
        task = asyncio.create_task(audio_emotion(filename))
        task_list.append(task)
    ans = await asyncio.gather(*task_list)
    print(len(ans))
    return ans






