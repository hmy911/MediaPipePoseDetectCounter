import os, sys, csv, json, time, copy, GPUtil, itertools, logging
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import Toplevel, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import google.generativeai as genai

import funcVideoDetect as fvd

# 读取 JSON 文件
def load_food_options():
    with open('food_options.json', 'r', encoding='utf-8') as file:
        return json.load(file)

#gemini ai
mygoogleapikey = 'AIzaSyAgaxfA7wunysfu6MBjikKDQ2txgh9DUow' # 替换API 密碼
genai.configure(api_key=mygoogleapikey)
modelai = genai.GenerativeModel("gemini-1.5-flash")

# 创建输出目录
output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model_path = 'my_lstm_model_1.h5' #模型1
model = fvd.load_model(model_path)

# 创建主窗口
root = tk.Tk()
root.title("卡路里計算器")
root.configure(background='#FDF7F7')

width = 1770
height = 970
left = 8
top = 3
root.geometry(f'{width}x{height}+{left}+{top}')

# 创建用于显示视频的 LabelFrame 和 Canvas
group_videos = tk.LabelFrame(root, padx=20, background='#99EEBB', fg='#000000', font=('Gabriola', 9, 'bold'), bd=2)
group_videos.grid(row=1, column=2, sticky='nsew')  # 布局，放置在网格的第2行，第0列
group_videos.grid_propagate(False)
group_videos.config(width=30, height=5)
tk.Label(group_videos, text='影像回顧', padx=20, bg='#99EEBB', fg='#000000', font=('Gabriola', 24, 'bold'), bd=2).grid(row=0, column=0, sticky='w')

# 创建 Canvas 用于显示视频
video_canvas = tk.Canvas(group_videos, width=450, height=300, bg='#FDF7F7')
video_canvas.grid(row=1, column=0)

# 定义全局变量
selected_date_var = tk.StringVar()
food_data = load_food_options()
dframe = 30


def save_to_csv():
    global selected_meals, selected_drinks 

    script_directory = os.path.dirname(os.path.abspath(__file__))
    csv_directory = os.path.join(script_directory, 'user_data')  # 在当前目录下创建 'cal' 文件夹
    json_file_path = os.path.join(csv_directory, 'user_data.json')  # JSON 文件路径
    csv_file_path = os.path.join(csv_directory, 'user_data.csv')

    # 创建一个字典并填充数据
    data = {
        'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'Name': [name_entry.get().strip()],
        'Age': [age_entry.get().strip()],
        'Height (cm)': [height_entry.get().strip()],
        'Weight (kg)': [weight_entry.get().strip()],
        'Gender': [gender_var.get().strip()],
        'Activity Level': [activity_level_var.get().strip()],
        'TDEE (kcal)': [tdee_var.get().strip()],
        'Activity  (kcal)': [total_var.get().strip()],
        'Activity1 Result (kcal)': [result1_var.get().strip()],
        'Food Cost (kcal)': [cost_var.get().strip()],
        'Remaining (kcal)': [less_var.get().strip()],
        'Selected Meals': [', '.join(selected_meals)],  # 假设 selected_meals 是全局变量
        'Selected Drinks': [', '.join(selected_drinks)]  # 假设 selected_drinks 是全局变量
    }

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    try:
        # 如果文件存在，追加数据；否则创建新文件
        if os.path.isfile(csv_file_path):
            df.to_csv(csv_file_path, mode='a', header=False, index=False, encoding='utf-8')
        else:
            df.to_csv(csv_file_path, mode='w', header=True, index=False, encoding='utf-8')

        # 将数据保存为 JSON 文件
        if os.path.isfile(json_file_path):
            # 如果 JSON 文件已存在，读取现有数据并追加新数据
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
                existing_data.append(data)  # 将新的数据追加到现有数据中

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(existing_data, json_file, ensure_ascii=False, indent=4)  # 确保中文正确显示
        else:
            # 创建新的 JSON 文件
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump([data], json_file, ensure_ascii=False, indent=4)  # 使用列表格式

        messagebox.showinfo("成功", "数据已成功保存到CSV和JSON文件")
        logging.info("用户数据已成功保存到 CSV 和 JSON 文件")

    except Exception as e:
        messagebox.showerror("錯誤", f"保存文件时发生错误: {e}")
        logging.error(f"保存文件时发生错误: {e}")

# 影片浏览位置
def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def show_chart():
    json_file = 'user_data.json'  # 定义要加载的JSON文件

    try:
        data = load_json_data(json_file)  # 尝试加载数据
    except FileNotFoundError:
        messagebox.showerror("错误", f"未找到文件: {json_file}")
        return

    # 创建新窗口以显示图表和数据
    chart_window = Toplevel(root)  # 确保chart_window在这里定义
    chart_window.title("歷史資料")  # 设置窗口标题

    # 获取所有用户的姓名，并去重后按字母顺序排序
    users = sorted(set(entry["Name"] for entry in data))

    # 用户选择标签和下拉菜单
    user_label = tk.Label(chart_window, text="選擇用戶:")
    user_label.pack(pady=5)
    
    selected_user_var = tk.StringVar()
    user_dropdown = tk.OptionMenu(chart_window, selected_user_var, *users, 
                                  command=lambda user: update_dates(data, user, date_dropdown, canvas, ai_response_label))
    user_dropdown.pack(pady=5)
    selected_user_var.set(users[0])  # 默认选择第一个用户

    # 日期选择标签
    date_label = tk.Label(chart_window, text="選擇日期:")
    date_label.pack(pady=5)

    selected_date_var = tk.StringVar()  # 绑定到 OptionMenu 的变量
    date_dropdown = tk.OptionMenu(chart_window, selected_date_var, '')  # 初始时没有日期选项
    date_dropdown.config(width=15) 
    date_dropdown.pack(pady=5)

    # 创建 Matplotlib 图表的 Canvas
    fig = plt.Figure(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # AI 建议标签
    ai_response_label = tk.Label(chart_window, text="", wraplength=400)
    ai_response_label.pack(pady=5)

    # 初始化日期和图表
    update_dates(data, users[0], date_dropdown, canvas, ai_response_label)  # 调用函数来更新日期和图表

# 读取 JSON 数据
def load_json_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 根据用户加载历史日期
def get_user_dates(data, selected_user):
    # 筛选出该用户的所有日期
    user_dates = [entry["Timestamp"].split(' ')[0] for entry in data if entry["Name"] == selected_user]
    return sorted(set(user_dates), reverse=True)

# 计算最近7天的平均卡路里消耗 (基于当前用户)
def calculate_7_day_average(data, selected_user, activity_names):
    # 筛选出该用户的数据，并按日期排序
    user_data = [entry for entry in data if entry["Name"] == selected_user]
    user_data = sorted(user_data, key=lambda x: x["Timestamp"], reverse=True)  # 按日期排序

    # 如果用户数据不足7天，取全部数据
    recent_data = user_data[:7] if len(user_data) >= 7 else user_data

    # 初始化每个活动的总卡路里
    activity_totals = {activity: [] for activity in activity_names}

    # 遍历最近7天的数据
    for entry in recent_data:
        activity_calories_str = entry["Activity1 Result (kcal)"]
        lines = activity_calories_str.split('\n')

        for line in lines:
            if line:
                parts = line.split(',')
                activity = parts[0].split(':')[0]
                calories = float(parts[1].split(' ')[1])  # 提取卡路里
                
                if activity in activity_totals:
                    activity_totals[activity].append(calories)

    # 计算每个活动的平均值
    activity_averages = {activity: np.mean(activity_totals[activity]) if activity_totals[activity] else 0
                         for activity in activity_names}
    return activity_averages

# 创建图表
def create_chart(entry, data, selected_user):
    activity_names = []
    activity_calories = []

    # 提取选择日期的活动数据
    activity_calories_str = entry["Activity1 Result (kcal)"]
    lines = activity_calories_str.split('\n')

    for line in lines:
        if line:
            parts = line.split(',')
            activity = parts[0].split(':')[0]
            calories = float(parts[1].split(' ')[1])  # 提取卡路里
            activity_names.append(activity)
            activity_calories.append(calories)

    # 计算该用户最近7天的平均卡路里消耗
    activity_averages = calculate_7_day_average(data, selected_user, activity_names)

    # 创建 Matplotlib 图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 创建柱状图，表示选择日期的卡路里消耗
    ax1.bar(activity_names, activity_calories, color='blue', label="Calories for Selected Date")
    ax1.set_xlabel('Activities')
    ax1.set_ylabel('Calories (kcal)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 确定y轴范围的最大值
    max_calories = max(max(activity_calories), max(activity_averages.values())) * 1.1

    # 统一y轴刻度范围
    ax1.set_ylim(0, max_calories)

    # 创建第二个坐标轴，用于绘制红色折线图，表示最近7天的平均卡路里消耗
    ax2 = ax1.twinx()  # 创建共享x轴的第二个坐标轴
    ax2.plot(activity_names, [activity_averages[activity] for activity in activity_names], color='red', marker='o', label='7-Day Average')
    ax2.set_ylabel('7-Day Avg Calories (kcal)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # 同步y轴范围
    ax2.set_ylim(0, max_calories)

    # 添加图表标题和图例
    ax1.set_title('Calories Burned by Activity (Selected Date vs. 7-Day Average)')
    fig.tight_layout()

    return fig

# 更新图表和显示AI建议
def update_chart_and_ai(data, selected_user, selected_date, canvas, ai_response_label):
    # 查找与所选日期和用户匹配的记录
    selected_entry = None
    for entry in data:
        if entry["Name"] == selected_user and entry["Timestamp"].startswith(selected_date):
            selected_entry = entry
            break

    if selected_entry:
        # 清除旧图表
        canvas.get_tk_widget().delete("all")  # 先清空Canvas
        
        # 创建并显示图表
        fig = create_chart(selected_entry, data, selected_user)  # 传入selected_user
        canvas.figure = fig
        canvas.draw()  # 重新绘制

        # 显示AI建议
        ai_response_text = selected_entry.get("AI Response", "無 AI 建議")
        ai_response_label.config(text=ai_response_text)
    else:
        # 如果未找到匹配的数据，清空显示
        ai_response_label.config(text="未找到匹配資料")

# 用户选择后更新日期选项
def update_dates(data, selected_user, date_dropdown, canvas, ai_response_label):
    # 获取该用户的所有日期
    user_dates = get_user_dates(data, selected_user)

    if user_dates:
        # 清空并更新日期下拉菜单
        date_dropdown['menu'].delete(0, 'end')
        for date in user_dates:
            date_dropdown['menu'].add_command(label=date, 
                                              command=lambda selected_date=date: update_chart_and_ai(data, selected_user, selected_date, canvas, ai_response_label))

        # 默认选择最新的日期
        latest_date = user_dates[0]  # 获取最新日期
        selected_date_var.set(user_dates[0])
        update_chart_and_ai(data, selected_user, user_dates[0], canvas, ai_response_label)
    else:
        # 如果没有可用的日期，清空下拉菜单
        date_dropdown['menu'].delete(0, 'end')
        date_dropdown['menu'].add_command(label="無可用日期", command=tk._setit(selected_date_var, "無可用日期"))
        ai_response_label.config(text="無可用日期")


# 计算BMR和TDEE的函数
def calculate_bmr_tdee():
    try:
        # 获取用户输入的年龄、身高和体重
        age = float(age_entry.get())
        height = float(height_entry.get())
        weight = float(weight_entry.get())
        gender = gender_var.get()
        activity_level = activity_level_var.get()

        # 计算BMR（基础代谢率）
        if gender == '男':
            bmr = (13.7 * weight) + (5 * height) - (6.8 * age) + 66
        else:
            bmr = (9.6 * weight) + (1.8 * height) - (4.7 * age) + 655

        # 活动水平的系数
        activity_multipliers = {
            "無活動：久坐": 1.2,
            "輕量活動：每周運動1-3天": 1.375,
            "中度活動量：每周運動3-5天": 1.55,
            "高度活動量：每周運動6-7天": 1.725,
            "非常高度活動量": 1.9
        }

                # 确保活动水平被正确选择
        if activity_level not in activity_multipliers:
            raise ValueError("請選擇有效的活動水平")
        
        # 计算TDEE（总每日能量消耗）
        tdee = bmr * activity_multipliers.get(activity_level, 1.9)

        # 显示计算结果
        tdee_var.set(f"{tdee:.2f} kcal")
        root.update()
        group2.update()

    except ValueError:
        # 错误处理：输入值无效
        messagebox.showerror("錯誤", "請輸入有效的數字")

calories_per_rep = {
        'pushup': 0.4,  # 伏地挺身每次0.4大卡
        "abworkout": 0.16,  # 仰臥起坐每次0.16大卡
        "squat": 0.42,  # 深蹲每次0.42大卡
        "pullup": 1.0,  # 拉單槓每次1大卡
        "run": 0.03,  # 跑步每步0.03大卡
        "jump": 0.19,  # 跳每次0.19大卡
        "rest": 0
    }

def calculate_calories(df):
    """計算每種運動的卡路里並返回結果字串。"""
    last_row = df.iloc[-1]
    calorie_results = ""
    total_calories = 0 

    for exercise, cal_per_rep in calories_per_rep.items():
        reps = last_row[exercise]  # 获取每种运动的次数
        calories = last_row[exercise] * cal_per_rep
        total_calories += calories  # 累加到总卡路里
        calorie_results += f"{exercise.capitalize()}: {reps}Time, {calories:.2f} kcal\n"

    calorie_results += f"\n總卡路里: {total_calories:.2f} kcal"
    return calorie_results,total_calories

# 食物计算函数
def calculate_food():
    global selected_meals, selected_drinks  # Declare them as global
    try:
        # 检查 tdee_var 是否有有效的值
        tdee_str = tdee_var.get().strip()
        print(f"DEBUG: TDEE value = '{tdee_str}'")  # 输出 TDEE 的实际内容
        
        # 从 TDEE 字符串中提取数值
        tdee_float = float(tdee_str.split()[0])  # 直接获取第一个元素（即数值）

        total_str = total_var.get()
        total_float = float(total_str)

        selected_meals = [meal for meal, var in meals_vars.items() if var.get()]
        selected_drinks = [drink for drink, var in drink_vars.items() if var.get()]

        cost = sum(food_data['meals'].get(meal, 0) for meal in selected_meals) + sum(food_data['drinks'].get(drink, 0) for drink in selected_drinks)

        total_have = tdee_float + total_float 
        less = total_have - cost

        cost_var.set(f"{cost:.2f} kcal")
        less_var.set(f"{less:.2f} kcal")
    except ValueError as e:
        messagebox.showerror("錯誤", f"請輸入有效的數字或檢查數值格式: {e}")

# 按钮点击事件处理函数
def cal_button_clicked_1():
    try:
        # 使用全局变量，而不是重新计算
        result1_var.set(calorie_results)

         # 计算食物并获取相关数据
        calculate_food()  # 确保在计算食物时已经保存数据

        # 调用保存到 CSV 的函数
        save_to_csv()
    except Exception as e:
        messagebox.showerror("錯誤", f"计算时发生错误: {e}")
        logging.error(f"计算时发生错误: {e}")

def get_gemini_response(input, image):
    if input != "":
        response = modelai.generate_content([input, image])
    else:
        response = modelai.generate_content(image)
    return response.text

def jsonDetail(jsonfile):
    """读取 JSON 文件并构建提示信息."""
    with open(jsonfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 从 JSON 文件中提取信息
    myprompt = f"{data}, 請以時間最近當日運動量推測每周 提供50字專業性熱量分析,運動及健康建議專業醫師回應 回應樣式:(姓名, 年紀, 青年中年男女)每日的基礎代謝率? 您的運動量每日推估?大卡 加上飲食?大卡 (夠或不夠) ，例如每周至少 (幾次?何種强度的 跑步或yoga 一周內 騎車 跑步幾分鐘? yoga幾分鐘?的組合)運動，每次至少 ? 分鐘。 早晨的營養均衡，但建議增加蛋白質攝取，例如雞蛋的優質蛋白，以提供更持久的能量。請用繁體中文回答"
    return myprompt

def save_response_to_json(filename, response):
    """将响应记录到 JSON 文件中的最近一条记录."""
    try:
        with open(filename, 'r+', encoding='utf-8') as f:
            data = json.load(f)

            # 检查 data 是否为列表且不为空
            if isinstance(data, list) and data:
                # 更新最后一条记录，添加 AI 响应
                data[-1]['AI Response'] = response

                # 移动文件指针到文件开头并写回修改后的数据
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=4)
                f.truncate()  # 截断文件以去除旧数据

            else:
                raise ValueError("JSON 文件不包含有效的数据列表")

    except Exception as e:
        print(f"保存 AI 响应时出错: {e}")

def ai_response():
    pathjson = "user_data.json"  # 请确保该 JSON 文件存在
    try:
        response = get_gemini_response(jsonDetail(pathjson), '')
        response_label.config(text=response)  # 将响应显示在 LabelFrame 中
        save_response_to_json(pathjson, response)
    except Exception as e:
        messagebox.showerror("错误", f"获取 AI 响应时发生错误: {e}")
    print(response)



###############################################################################
#Tkinter 介面
###############################################################################

tk.Label(root, text='AI 熱量管理師', background='#E99771',fg = '#000000', font=('Gabriola',36,'bold'),bd = 2  ).grid(row=0, column=0, sticky='nsew')

group0 = tk.Label(root, padx=10, pady=10, background='#E99771',fg = '#000000', font=('Gabriola',30,'bold'),bd = 2 )
group0.grid(row=2, column=0, sticky='nsew')
group0.grid_propagate(False)
group0.config(width=25, height=9)

tk.Label(group0, text='''這是一個卡路里計算器\n
此應用程序將幫助您計算以下內容：\n
- 總能量消耗 (TDEE)\n
- 依據運動和活動類別計算消耗的卡路里\n
- 根據選擇的食物計算攝取的卡路里\n
- 記錄歷史資料\n
- 提供AI飲食運動建議\n
運動強度可參考網站:https://reurl.cc/xv6k9V''', 
                        justify = 'left',bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=0, column=0, sticky='w')

group1= tk.Label(root, padx=20, pady=10, background='#E99771',fg = '#000000', font=('Gabriola',30,'bold') , bd = 2 )
group1.grid(row=1, column=0, sticky='nsew')
group1.grid_propagate(False)
group1.config(width=25, height=5)


tk.Label(group1, text='基本資料', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',24,'bold'), bd = 2 ).grid(row=0, column=0, sticky='w')
tk.Label(group1, text='姓名', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold'), bd = 2 ).grid(row=1, column=0, sticky='w')
name_entry = tk.Entry(group1)
name_entry.insert(0, "王小明")  # 预设值
name_entry.grid(row=1, column=1, sticky='w')
tk.Label(group1, text="年齡(歲):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=2, column=0, sticky='w')
age_entry = tk.Entry(group1)
age_entry.insert(0, "25")  # 预设值
age_entry.grid(row=2, column=1, sticky='w')

tk.Label(group1, text="身高(公分):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=3, column=0, sticky='w')
height_entry = tk.Entry(group1)
height_entry.insert(0, "170")  # 预设值
height_entry.grid(row=3, column=1, sticky='w')

tk.Label(group1, text="體重(公斤):", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=4, column=0, sticky='w')
weight_entry = tk.Entry(group1)
weight_entry.insert(0, "70")  # 预设值
weight_entry.grid(row=4, column=1, sticky='w')

tk.Label(group1, text="性別:", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=5, column=0, sticky='w')
gender_var = tk.StringVar(value="男")
tk.Radiobutton(group1, text="男", bg = '#E99771',fg = '#000000', font= ('Gabriola',12,'bold'), variable=gender_var, value="男",anchor = 'w').grid(row=5, column=1, sticky = 'w')
tk.Radiobutton(group1, text="女", bg = '#E99771',fg = '#000000', font= ('Gabriola',12,'bold'), variable=gender_var, value="女",anchor = 'w').grid(row=5, column=2, sticky = 'w')

tk.Label(group1, text="活動量:", bg = '#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold')).grid(row=6, column=0, sticky='w')
activity_level_var = tk.StringVar(value="無活動：久坐")
activity_level_menu = tk.OptionMenu(group1, activity_level_var, "無活動：久坐", "輕量活動：每周運動1-3天", "中度活動量：每周運動3-5天", "高度活動量：每周運動6-7天", "非常高度活動量") 
activity_level_menu.config  (bg = '#E99771',fg = '#000000', font=('Gabriola',12,'bold'))
activity_level_menu.grid(row=6, column=1, sticky='w')
tk.Label(group1, text='       ', background='#E99771',fg = '#000000', padx=30, font=('Gabriola',12,'bold'), bd = 2 ).grid(row=7, column=0, sticky='w')
tk.Button(group1, text="計算 TDEE", bg='#000000', fg='#ffffff', font=('Gabriola',10, 'bold'), command=calculate_bmr_tdee).grid(row=8, column=0, columnspan=2)

top1= tk.Label(root, background='#FDF7F7',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2  )
top1.grid(row=0, column=1, sticky='nsew')
top1.grid_propagate(False)
top1.config(width=40, height=2)

group2= tk.Label(top1, background='#EBD663',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2 , anchor = 'w' )
group2.grid(row=0, column=0)
group2.grid_propagate(False)
group2.config(width=20, height=2)
tk.Label(group2, text="TDEE", bg='#EBD663', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0,  columnspan=2)
tdee_var = tk.StringVar(value="")
tk.Label(group2, textvariable=tdee_var, bg='#EBD663', fg='#000000', padx=15, font=('Gabriola', 24, 'bold')).grid(row=1, column=1, sticky='nsew')

group2_1= tk.Label(top1, text="+", background='#FDF7F7',fg = '#000000', font=('Gabriola',12,'bold'), bd = 2  )
group2_1.grid(row=0, column=1)
group2_1.grid_propagate(False)
group2_1.config(width=1, height=2)

total_var = tk.StringVar()

def display_results(calorie_results, total_calories):
    group3= tk.Label(top1, background='#629677',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2 , anchor = 'e')
    group3.grid(row=0, column=2)
    group3.grid_propagate(False)
    group3.config(width=20, height=2)
    tk.Label(group3, text="運動消耗熱量", bg='#629677', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0, columnspan=2)

    """顯示卡路里結果的 Tkinter 窗口，並放置在 LabelFrame 內。"""
    # 创建 LabelFrame
    result_frame = tk.LabelFrame(group6, text="運動細項", padx=10, pady=10, font=('Gabriola', 12, 'bold'))
    result_frame.grid(row=2, column=0, sticky='w')
    
    # 在 LabelFrame 中添加结果标签
    result_label = tk.Label(result_frame, text=calorie_results, justify=tk.LEFT)  # 左对齐
    result_label.pack()  # 将标签放入 LabelFrame
    total_label = tk.Label(group3, text = total_calories, bg='#629677', fg='#000000', font=('Gabriola', 20, 'bold'))
    total_label.grid(row=1, column=1, sticky='nsew')

def cal_button_clicked():
    global calorie_results, total_calories, root
    try:
        video1 = video1_entry.get()
        # 計算卡路里
        # calorie_results, total_calories = fvd.play_video_with_landmarks_and_calories(video1, model, 30 )
        # calorie_results, total_calories = fvd.play_video_with_landmarks_and_calories(video1, model, 30, video_canvas, root, mp_pose, mp_drawing, output_dir )
        getdf = fvd.play_video_with_landmarks_and_calories(video1, model, 30, video_canvas, root, mp_pose, mp_drawing, output_dir )
   
        calorie_results, total_calories = calculate_calories(getdf)
        display_results(calorie_results,total_calories)
        # return calorie_results, total_calories

        result1_var.set(calorie_results)
        total_var.set(total_calories)
        df = pd.read_csv('output_data.csv')
        
    except Exception as e:
        messagebox.showerror("錯誤", f"計算時發生錯誤: {e}")
        logging.error(f"計算時發生錯誤: {e}")


top2= tk.Label(root, background='#FDF7F7',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2  )
top2.grid(row=0, column=2, sticky='nsew')
top2.grid_propagate(False)
top2.config(width=40, height=2)

group4_2= tk.Label(top2, text="-", background='#FDF7F7',fg = '#000000', font=('Gabriola',18,'bold'), bd = 2  )
group4_2.grid(row=0, column=0)
group4_2.grid_propagate(False)
group4_2.config(width=1, height=2)

group4= tk.Label(top2, background='#34AAD1',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2  )
group4.grid(row=0, column=1)
group4.grid_propagate(False)
group4.config(width=20, height=2)
tk.Label(group4, text="攝取熱量", bg='#34AAD1', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0,  columnspan=2)
cost_var = tk.StringVar(value="")
tk.Label(group4, textvariable=cost_var, bg='#34AAD1', fg='#000000', font=('Gabriola', 24, 'bold')).grid(row=1, column=1, sticky='nsew')

group4_1= tk.Label(top2, text="=", background='#FDF7F7',fg = '#000000', font=('Gabriola',18,'bold'), bd = 2  )
group4_1.grid(row=0, column=2)
group4_1.grid_propagate(False)
group4_1.config(width=1, height=2)

group5= tk.Label(top2, background='#2589BD',fg = '#000000', font=('Gabriola',28,'bold'), bd = 2 )
group5.grid(row=0, column=3)
group5.grid_propagate(False)
group5.config(width=20, height=2)
tk.Label(group5, text="剩餘", bg='#2589BD', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0, columnspan=2)
less_var = tk.StringVar(value="")
tk.Label(group5, textvariable=less_var, bg='#2589BD', fg='#000000', font=('Gabriola', 24, 'bold')).grid(row=1, column=1, sticky='nsew')

# 運動
group6= tk.LabelFrame(root, padx=20, background='#99EEBB',fg = '#000000', font=('Gabriola',30,'bold'), bd = 2  )
group6.grid(row=1, column=1, sticky='nsew')
group6.grid_propagate(False)
group6.config(width=30, height=5)

tk.Label(group6, text="今日運動:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0, sticky='w')
tk.Label(group6, text="將運動影片上傳進行分析運動類別及熱量追蹤", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 16, 'bold')).grid(row=1, column=0, sticky='w')
tk.Label(group6, text="影片上傳:", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 14, 'bold')).grid(row=2, column=0, sticky='w')
video1_entry = tk.Entry(group6)
video1_entry.grid(row=3, column=1)
tk.Button(group6, text="瀏覽", command=lambda: browse_file(video1_entry), bg = '#99EEBB',fg = '#000000', font=('Gabriola',9,'bold')).grid(row=2, column=1)
result1_var = tk.StringVar()
tk.Entry(textvariable=result1_var).grid(row=3, column=0)
tk.Label(group6, text="    ", bg='#99EEBB', fg='#000000', padx=30, font=('Gabriola', 12, 'bold')).grid(row=4, column=0, sticky='w')
tk.Button(group6, text="計算", command=cal_button_clicked, bg = '#99EEBB',fg = '#000000', font=('Gabriola',12,'bold')).grid(row=5, column=1)

#食物
group7= tk.LabelFrame(root, padx=20, background='#5ED7FF', width=30, height=8,fg = '#000000', font=('Gabriola',30,'bold'), bd = 2 )
group7.grid(row=2, column=1, sticky='nsew')
group7.grid_propagate(False)
group7.config(width=30, height=3)

tk.Label(group7, text="用餐選擇", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 24, 'bold')).grid(row=0, column=0, sticky='w')
tk.Label(group7, text="幫你計算早餐攝取多少熱量", bg='#5ED7FF', padx=30, fg='#000000', font=('Gabriola', 16, 'bold')).grid(row=1, column=0, sticky='w')

# 主餐部分
group7_1 = tk.LabelFrame(group7, text='主餐', padx=20, pady=10,  background='#32B6FD', fg='#000000', font=('Arial',14, 'bold'), bd=2)
group7_1.grid(row=2, column=0, padx=20, sticky='w')

meals_vars = {}  # 存储主餐 Checkbutton 的变量
row = 0  # 初始化行号
col = 0  # 初始化列号
for meals in food_data["meals"]:
    var = tk.BooleanVar()
    # 使用grid布局，每行显示7个Checkbutton
    tk.Checkbutton(group7_1, text=meals, variable=var, bg='#32B6FD', fg='#000000', font=('Arial', 10)).grid(row=row, column=col, sticky='w')
    meals_vars[meals] = var
    col += 1  # 列号加1

    # 如果列号达到7，换行，列号归零
    if col == 5:
        col = 0
        row += 1

tk.Label(group7, text="    ", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 5, 'bold')).grid(row=3, column=0, sticky='w')

# 饮料部分
group7_2 = tk.LabelFrame(group7, text='飲品', padx=20, pady=10, background='#32B6FD', fg='#000000', font=('Arial', 14, 'bold'), bd=2)
group7_2.grid(row=4, column=0, padx=20, sticky='w')

drink_vars = {}
row = 0  # 初始化行号
col = 0  # 初始化列号
for drink in food_data["drinks"]:
    var = tk.BooleanVar()
    # 使用grid布局，每行显示7个Checkbutton
    tk.Checkbutton(group7_2, text=drink, variable=var, bg='#32B6FD', fg='#000000', font=('Arial', 10)).grid(row=row, column=col, sticky='w')
    drink_vars[drink] = var
    col += 1  # 列号加1

    # 如果列号达到7，换行，列号归零
    if col == 4:
        col = 0
        row += 1

tk.Label(group7, text="    ", bg='#5ED7FF', fg='#000000', padx=30, font=('Gabriola', 5, 'bold')).grid(row=5, column=0, sticky='w')
tk.Button(group7, text="計算熱量",padx=20, pady=10, bg='#32B6FD', fg='#000000', font=('Arial', 12, 'bold'), command=calculate_food).grid(row=6, column=0, columnspan=2)


# 创建一个 LabelFrame 作为 group8
group8 = tk.LabelFrame(root, padx=20, background='#A684C2', fg='#000000', font=('Gabriola', 30, 'bold'), bd=2)
group8.grid(row=2, column=2, sticky='nsew')
group8.grid_propagate(False)  # 防止 group8 自动调整大小
group8.config(width=30, height=4)  # 设置固定的宽度和高度
group8_1 = tk.Label(group8, background='#A684C2', fg='#000000', font=('Gabriola', 30, 'bold'))
group8_1.grid(row=1, column=0, sticky='w')
# 在 group8 中添加 "建議" 标签
suggestion_label = tk.Label(group8, text="建議", padx=20, background='#A684C2', fg='#000000', font=('Gabriola', 24, 'bold'))
suggestion_label.grid(row=0, column=0, sticky='w')

# 在 group8 中创建显示图表的按钮

chart_button = tk.Button(group8_1, text="顯示圖表", command=show_chart, bg='#654F6F', fg='#000000', font=('Gabriola', 12, 'bold'))
chart_button.grid(row=0, column=1)  # 使用 padx 和 pady 确保按钮有足够的空间
calculate_button = tk.Button(group8_1, text="保存資料", command=cal_button_clicked_1, bg='#654F6F', fg='#000000', font=('Gabriola', 12, 'bold'))
calculate_button.grid(row=0, column=0)  # 使用 pack() 方法布局，添加适当的间距
get_response_button = tk.Button(group8_1, text="AI建議", command=ai_response,  bg='#654F6F', fg='#000000',font=('Gabriola', 12, 'bold'))
get_response_button.grid(row=0, column=2)

response_frame = tk.LabelFrame(group8, text="AI 建議", padx=10, pady=10, bg='#654F6F', fg='#000000', font=('Gabriola', 12, 'bold'))
response_frame.grid(row=2, column=0)
response_frame.config(width=70, height=8)

response_label = tk.Label(response_frame, text="", padx=10, pady=10, wraplength=500, justify="left", fg='#000000', font=('Gabriola', 13, 'bold'))
response_label.grid(row=0, column=0)
response_label.config(width=66, height=7)

root.mainloop()
