import asyncio
import os

from Main_agent import create_main_agent
agent=create_main_agent()

job = """"""
path = ''
async def main1(): # 简历评价
    inputs = {'path': path, 'workflow_step': '简历评价', 'job': job}
    thread_config = {"configurable": {"thread_id": "5"}}
    async for chunk in agent.astream(inputs, thread_config, stream_mode="updates"):
        print("--- CHUNK ---")
        print(chunk)
        print("\n")

async def main2(): # 简历优化
    inputs = {'path': path, 'workflow_step': '简历优化', 'job': job}
    thread_config = {"configurable": {"thread_id": "5"}}
    async for chunk in agent.astream(inputs, thread_config, stream_mode="updates"):
        print("--- CHUNK ---")
        print(chunk)
        print("\n")
        if '__interrupt__' in chunk:
            print('简历需要用到本人照片，如果您方便提供请给出路径：\n')
            data = input('请输入照片路径（如果您不想给出那么直接回车即可，会用默认的图片生成优化的简历）')
            agent.update_state(
                thread_config,
                {"picture_path": data}
            )
            async for chunk in agent.astream(None, thread_config, stream_mode="updates"):
                print("--- CHUNK ---")
                print(chunk)
                print("\n")

async def main3(): # 面试训练
    inputs = {'path': path, 'workflow_step': '面试训练', 'job': job,'interview_question_num':[1,0,0,0]}#可以随意更改数目
    thread_config = {"configurable": {"thread_id": "5"}}
    # FIX: Change stream_mode from "messages" to "updates"
    async for chunk in agent.astream(inputs, thread_config, stream_mode="updates"):
        print("--- CHUNK ---")
        print(chunk)
        print("\n")


async def main4(): # 模拟面试
    inputs = {'path': path, 'workflow_step': '算法测试', 'job': job,'code_id':'zhangminghao'}#需要给出你在codeforce对应的账号名字
    thread_config = {"configurable": {"thread_id": "5"}}
    # FIX: Change stream_mode from "messages" to "updates"
    async for chunk in agent.astream(inputs, thread_config, stream_mode="updates"):
        print("--- CHUNK ---")
        print(chunk)
        print("\n")

if __name__ == "__main__":
    directory_names_to_create = [
        "pdf_reports",
        "resume_data",
        "video_picture",
        "简历评估",
        "简历照片",
        "雷达图",
        "面试视频（用户）",
        "面试知识库",
        "问题解析",
        "优化简历",
        "语音资料（用户）",
    ]
    for directory_name in directory_names_to_create:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    # asyncio.run(main4())
