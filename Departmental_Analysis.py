#大部分和data_process相同
import colorsys
import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import random
import matplotlib as mpl
import numpy as np

# import warnings
# warnings.filterwarnings("ignore")
class init_analysis():#默认信息类，用于默认初始化dataframe信息
    FILE_ADDR = "Employee_Salaries.csv"         #输入文件名
    OUTPUT_ADDR = "end.csv"                     #输出文件名
    LABEL_TUPLE = ['Department', 'Department_Name', 'Division', 'Gender', 'Base_Salary',
       'Overtime_Pay', 'Longevity_Pay', 'Grade']    #csv的行标签元素
    MODEL_CAL_MORE = 1  #指代运算符模式，1为大于、-1为小于、0为等于
    MODEL_CAL_LESS = -1
    MODEL_CAL_EQUAL = 0
    SELECT_DICT = {'Department': [MODEL_CAL_EQUAL, 'ABS']} #默认筛选样本的字典
    CAL_DEFAULT_LIST = ['mean', 'std', 'min', '50%']#'count', 'mean', 'std', 'min', 'max', '50%'
    DEBUG_MODEL = 1 #调试模式
    INFO = 1
    ERROR = -1
    WARNING = 0
def print_log(info:str, model:int = init_analysis.INFO):
    if model:
        print("[info]"+info)
    elif not model:
        print("[error]"+info)
    else:
        print("[warn]"+info)

def load_and_init_dataSet(file_address:str = init_analysis.FILE_ADDR ):
    """
    构造初始化的数据集
    :param file_address: 文件地址【带后缀】
    :return: data_ori 返回构造后的数据集dataframe
    """
    data_ori = None
    if osp.isfile(file_address):
        data_ori = pd.read_csv(file_address)
        print("[info] data load end")
        print(data_ori.info())
    else:
        print("[warn] file not find error,try open default")
        if osp.isfile(init_analysis.FILE_ADDR):
            data_ori = pd.read_csv(init_analysis.FILE_ADDR)
            print("[info] data load end")
        else:
            print("[error] csv read error")
    return data_ori

def select_some_label(data_ori:pd.DataFrame, select_dict=None):
    """
    通过筛选 选择字典划分目标数据集【没写查重处理，懒得写了】
    :param data_ori: 输入的原始数据集
    :param select_dict: 筛选的字典变量
    :return: data_select 筛选后的数据集变量DataFrame
    """
    if select_dict is None:#如果无选择则选择默认选择
        print("[warn] select dict is none, use default dict")
        select_dict = init_analysis.SELECT_DICT
    data_select = data_ori
    for index, value_list in select_dict.items():#遍历筛选字典
        cal_label = value_list[0]
        value = value_list[1]
        if not index in init_analysis.LABEL_TUPLE:#如果标签不符合直接返回error-1
            print("[error] label lawlessness")
            return -1
        elif cal_label == init_analysis.MODEL_CAL_EQUAL:#通过上文构造的MODEL_CAL_ 算术符号做处理
            data_select = data_select[data_select[index] == value]
        elif cal_label == init_analysis.MODEL_CAL_MORE:
            data_select = data_select[data_select[index] > value]
        elif cal_label == init_analysis.MODEL_CAL_LESS:
            data_select = data_select[data_select[index] < value]

    # print(data_select.head())
    return data_select

def save_dataFarme(data_end :pd.DataFrame ,file_address:str = init_analysis.OUTPUT_ADDR, file_index=False):
    if not osp.isfile(file_address):
        print("[warn] output file not found, try create new file")
        f_new = open(file_address, "w+", encoding="UTF-8")
        f_new.close()
    if '.csv' in file_address :
        data_end.to_csv(file_address, index=file_index)
    elif ".xlsx" in file_address:
        data_end.to_excel(file_address, index=file_index)
    print(f"[info] data save {file_address} process end")
    return 1

def plot_with_dataframe(data:pd.DataFrame,list_info:list ,model = 'default',save_model:bool = False, save_pic_addr:str = None):
    """
    对describe的dataframe进行可视化
    :param data: 输入的describe处理后的数据
    :param list_info: x轴标签和标题
    :param model: 模式【str字符串选择】
    :param save_model: 保存模式，默认不保存
    :param save_pic_addr: 保存文件路径
    :return:
    """

    if list_info == None:
        print("[error] info list is empty")
        return

    if model == 'default':
        fig, axs = plt.subplots(1, 4, figsize=(16, 8))
        sns.lineplot(data, ax=axs[0])
        sns.boxplot(data, ax=axs[1])
        sns.barplot(data, errorbar=None, ax=axs[2])
        sns.pointplot(data, ax=axs[3])
        plt.tight_layout()
        if save_model and save_pic_addr:
            plt.savefig(save_pic_addr)
            plt.clf()
        else:
            plt.show()
            plt.clf()

        return
    flg , ax1 = plt.subplots()
    if model == 'box':
        sns.boxplot(data)
    elif model == 'bar':
        sns.barplot(data, errorbar=None)
    elif model == 'point':
        sns.pointplot(data)
    elif model == 'line':
        sns.lineplot(data)
    plt.xlabel(list_info[1])
    plt.title(list_info[0])
    plt.show()
    return

def set_plot_label_and_title(axs, base_list:list, type_plot:list):
    title, xlabel, ylabel = base_list[0], base_list[1], base_list[2]
    len_axs = len(axs)
    for i in range(len_axs):
        axs[i].set_xlabel(xlabel+type_plot[i])
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(title+type_plot[i])
    return axs
def cal_describe(data_select:pd.DataFrame, default_model:list = None):
    res_data = data_select.describe()
    res_data = res_data.T
    if default_model != None:
        res_data = res_data[default_model]
    if init_analysis.DEBUG_MODEL:
        print(res_data)
    return res_data.T
def count_size_with_data(data_sorted:pd.DataFrame, label:str,start, end ):
    """
    获取符合【start,end】区间的个数
    :param data_sorted: 数据集
    :param label: 标签
    :param start: 开始的value
    :param end: 结束的value
    :return: 符合要求的个数
    """
    res_list =  (data_sorted[label] > start) & (data_sorted[label] < end)
    res = data_sorted[ res_list ]
    return len(res)
def statistics_sorted_data(data_sorted:pd.DataFrame, label:str,  batch:int=5000):
    """
    统计data_sorted的分布信息图并返回字典【单变量】,且对应的标签必须是可计算的，懒得写判断了（之间调用sns太慢了，还容易溢出，自己写一个，而且还能保存这个数据）
    :param data_sorted: 输入已从小到大排布的数据
    :param label: 选定的筛选主标签
    :param batch: 批次【按照这个划分】
    :return:res_set 字典类型的分布序列
    """
    start = data_sorted.min()
    start = int(start[label]) // 100
    start *= 100
    end = data_sorted.max() // 1000
    end *= 1001
    end = int(end[label])# 获取最大最小值区间，并取整

    res_set = {}
    for size_p in range(start, end, batch):
        temp_set = {start:count_size_with_data(data_sorted, label, start, start+batch)}
        res_set.update(temp_set)
        start += batch
    print(f"[info] statistics success")
    return res_set
def plot_one_label_histogram(data_ori:pd.DataFrame, label:str, save_pic_addr:str = None ,save_model = False, set_color_gradient:bool = False):
    """
    对上述statistics_sorted_data的一键傻瓜化，之间调用api即可实现->获取单个标签在数据集中的直方图
    步骤如下：1.获取单个标签信息 2.排序【从小到大】 3.调用statistics_sorted_data返回分布信息字典 4.对图像进行后处理操作
    :param data_ori:输入数据集
    :param label: 标签信息【只能是单个标签】
    :param save_pic_addr: 保存文件路径【非默认】
    :param save_model: 保存模式，默认是false，如果是true且路径ok就保存
    :param set_color_gradient:渐变模式
    """
    plt.figure(figsize=(12, 6))
    temp = data_ori[[label]]
    # print(f"[info]ready sort by {label} value")
    sort_temp = temp.sort_values(by=label)
    print(f"[info] building {label} histogram ")
    statis_data = statistics_sorted_data(sort_temp, label)

    statis_data = pd.DataFrame(statis_data, index=[0])
    # print(f"[info] success sort by {label} value")
    if set_color_gradient :
        sns.barplot(statis_data, color='#7b68ee')
    else:
        sns.barplot(statis_data)
    plt.xticks(rotation=300)
    plt.tight_layout()
    plt.title(label+"label histogram")
    plt.xlabel("salary")
    plt.ylabel("count of "+label)
    if save_model == True and save_pic_addr:
        plt.savefig(save_pic_addr,bbox_inches='tight')
        print(f"[info] save {label} histogram in {save_pic_addr}")
        plt.clf()
        return
    else:
        plt.show()
        print(f"[info] build {label} histogram success")
        return

def plot_between_two_data(data_1:pd.DataFrame, data_2:pd.DataFrame, save_jpg_addr:str = None, save_model:bool = False):
    """
    将两个数据集【两个都是经过cal_describe处理的】的三个基本工资通过一张图展示出来,如果需要保存则保存，默认不保存
    :param data_1:
    :param data_2:
    :param save_jpg_addr:
    :param save_model:
    :return:
    """
    hue1 = ['Category A', 'Category B', 'Category C', 'Category D']
    fig, ax = plt.subplots(3, 2)
    data_select_M = data_1
    data_select_F = data_2
    sns.barplot(data_select_M['Base_Salary'], ax=ax[0, 0], palette='bright')
    sns.barplot(data_select_M['Overtime_Pay'], ax=ax[1, 0], palette='Set3')
    sns.barplot(data_select_M["Longevity_Pay"], ax=ax[2, 0], palette='pastel')  # 对不同像素图片进行颜色、坐标轴等信息修改，下同理

    for i in range(2):
        ax[i, 0].set_ylabel("Salary")
    ax[0, 0].set_xlabel("Male Base_Salary")
    ax[1, 0].set_xlabel("Male Overtime_Pay")
    ax[2, 0].set_xlabel("Male Longevity_Pay")

    sns.barplot(data_select_F['Base_Salary'], ax=ax[0, 1], palette='bright')#color="#00BFFF",
    sns.barplot(data_select_F['Overtime_Pay'], ax=ax[1, 1], palette='Set3')
    sns.barplot(data_select_F["Longevity_Pay"], ax=ax[2, 1], palette='pastel')
    for i in range(3):
        ax[i, 1].set_ylabel("Salary")
    ax[1, 1].set_ylim(0, 20000)  # 设置坐标轴统一
    ax[2, 1].set_ylim(0, 4000)
    ax[0, 1].set_xlabel("Female Base_Salary")
    ax[1, 1].set_xlabel("Female Overtime_Pay")
    ax[2, 1].set_xlabel("Female Longevity_Pay")
    plt.tight_layout()
    if save_model and ".jpg" in save_jpg_addr :
        plt.savefig(save_jpg_addr, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

def plot_with_n_scatter(data_ori_list :list, pick_label_list:list, save_model:bool = False, save_jpg_addr:str = None):
    """
    对公司和薪资画散点图
    :param data_ori_list:  公司数据集列表
    :param pick_label_list: 筛选标签【必须是两个值，散点图只能接收两个】 ex:["Base_Salary","Department"]
    :param save_model:保存模式
    :param save_jpg_addr: 保存地址
    """
    if init_analysis.DEBUG_MODEL:
        print_log("start plot scatter")
    fig, ax = plt.subplots( figsize=(12,12) )
    if len(pick_label_list) != 2:
        print_log("label list len unequal 2", init_analysis.ERROR)
        return
    list_len = len(data_ori_list)
    colors = mpl.cm.rainbow( np.arange(list_len) / list_len)
    for index, data_ori in enumerate(data_ori_list):
        data_selected = data_ori[pick_label_list]
        sns.stripplot(data_selected, y="Department", x=pick_label_list[0], color=colors[index])
    plt.tight_layout()
    if save_model and save_jpg_addr:
        plt.savefig(save_jpg_addr+""+pick_label_list[0])
        plt.clf()
    else:
        print_log("success plot but not save")
        plt.clf()
    if init_analysis.DEBUG_MODEL:
        print_log("success plot scatter")
def plot_pie_chart(nums:list, label:list,title:str = None, save_model:bool = False, save_addr = None):

    """
    对原饼图进行修改 不需要使用随机生成颜色，设置了文字的大小使用归一化的连续颜色来表示
    :param nums: 输入的数量count列表
    :param label: 输入的标签列表
    :param title: 标题
    :param save_model: 保存模式
    :param save_addr: 保存图片的地址
    """
    threshold = 1.2#不显示的阈值【百分比】
    def my_autopct(pct):
        return f'{pct:.1f}%' if pct >= threshold else None

    fig, ax = plt.subplots(figsize=(8,8))
    if len(nums) != len(label):
        print("[error] nums len != label len")
        return
    len_n = len(nums)

    colors = mpl.cm.rainbow(np.arange(len_n) / len_n)

    # patches, texts, autotexts = ax.pie(nums, labels=label, autopct='%0.1f%%',startangle=100,colors=colors)
    patches, texts, autotexts = ax.pie(nums, labels=label, autopct=my_autopct, startangle=100, colors=colors)

    proptease = fm.FontProperties()
    proptease.set_size('x-small')

    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)

    ax.axis('equal')
    plt.title(title)
    if save_model and save_addr:
        plt.savefig(save_addr)
        plt.clf()
    else:
        plt.show()
        plt.clf()
    return

def plot_bar_with_n(data_describe_list:list, pick_label_list:list, company_list:list, select_what_salary:str = "Base_Salary", color_set:str = "	#8A2BE2",save_addr:str = None):
    """
    画出公司之间salary variation的条形图，可用指定多个参数，见下文
    :param data_describe_list: 带有多个公司数据集的列表【已describe】
    :param pick_label_list: 选取的算术标签列表【在describe的标签】
    :param company_list: 带公司名称的列表
    :param select_what_salary: 对哪一个薪资进行分析【默认为base salary】
    :param color_set: 设定颜色【16进制输入】
    :param save_addr: 保存图片地址，默认不保存
    """

    data_dict_new = dict()
    for index, data_ori in enumerate(data_describe_list):
        data_payed = data_ori[select_what_salary]
        data_payed = data_payed[pick_label_list]
        company_name = company_list[index]
        temp_dict = {}
        for index_label , cal_label in enumerate(pick_label_list):

            if not index:
                firm_list = list()
                if not index_label:
                    firm_list.append(company_name)
                    temp_dict.update({"company_name": firm_list})
                add_num = list()
                add_num.append(data_payed[cal_label])
                temp_dict.update( {cal_label: add_num }
                                  )
                data_dict_new.update(temp_dict)
            else:
                data_dict_new[cal_label].append(data_payed[cal_label])
                if not index_label:
                    data_dict_new["company_name"].append(company_name)


    for index, cal_label in enumerate(pick_label_list):
        if init_analysis.DEBUG_MODEL:
            print_log("start plotting bar")
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.barh(data_dict_new["company_name"], data_dict_new[cal_label], color="#FF8C00")
        plt.title(f"company {select_what_salary} analysis("+cal_label+")")
        plt.xlabel("Salary")
        if save_addr:
            plt.savefig(save_addr+select_what_salary+"_"+cal_label)
            plt.clf()
            if init_analysis.DEBUG_MODEL:
                print_log(f"success plotting {cal_label} bar in {save_addr} ")
        else:
            plt.show()
            plt.clf()

    return
def department_data_analysis(data_ori:pd.DataFrame, out_min:int = 20):


    department_dataframe_list = list()
    department_describe_list = list()

    department_label_list = data_ori["Department"].tolist()#职位列表标签
    label_key = set(department_label_list)
    labels_ori = list(label_key)#去重处理

    label_and_count_set = dict()
    #构造公司{名称：人员数量}字典，自动去除小于最小阈值out_min的公司
    for temp_key in labels_ori:
        if department_label_list.count(temp_key) >= out_min:
            temp_set = {temp_key : department_label_list.count(temp_key)}
            label_and_count_set.update(temp_set)
    #从小到大排序公司样本数
    label_and_count_set = dict(sorted( label_and_count_set.items(),key=lambda x:x[1] ) )
    #字典的key是公司名，value是人数
    print_log("label count:")
    print( label_and_count_set.keys())
    print_log("label list:")
    print( label_and_count_set.values())
    plot_pie_chart(list(label_and_count_set.values()), label=list(label_and_count_set.keys()), title="department_rate",
                   save_model=True, save_addr="save_file/Department_pic/all_departments_rate")

    #对每个公司对应的数据集进行划分，保存到department_dataframe_list中，处理后数据放入department_dataframe_list中，一一对应
    for index, department_name in enumerate(label_and_count_set.keys() ):
        department_dataframe_list.append(
            select_some_label(data_ori=data_ori, select_dict={"Department": [init_analysis.MODEL_CAL_EQUAL, department_name]})
            )
        department_describe_list.append(
            cal_describe(department_dataframe_list[index], init_analysis.CAL_DEFAULT_LIST)
        )
    # 对薪资数据【Base_Salary、Overtime_Pay、Longevity_Pay】进行可视化并保存
    for index, department_name in enumerate(label_and_count_set.keys() ):
        print_log(f"index {index} in {department_name} plot:")
        plot_with_dataframe(department_describe_list[index],["salary", department_name],save_model=True,
                            save_pic_addr= "save_file/Department_pic/description/"+department_name+"_description")
        plot_one_label_histogram(department_dataframe_list[index], "Base_Salary", "save_file/Department_pic/Base_Salary/"+department_name, save_model=True, set_color_gradient=True)
        plot_one_label_histogram(department_dataframe_list[index], "Overtime_Pay", "save_file/Department_pic/Overtime_Pay/"+department_name, save_model=True, set_color_gradient=True)
        plot_one_label_histogram(department_dataframe_list[index], "Longevity_Pay", "save_file/Department_pic/Longevity_Pay/"+department_name, save_model=True, set_color_gradient=True)

    plot_with_n_scatter(department_dataframe_list, ["Base_Salary","Department"],save_model=True,
                        save_jpg_addr="save_file/Department_pic/Base_Salary/")
    plot_with_n_scatter(department_dataframe_list, ["Overtime_Pay", "Department"], save_model=True,
                        save_jpg_addr="save_file/Department_pic/Overtime_Pay/")
    plot_with_n_scatter(department_dataframe_list, ["Longevity_Pay", "Department"], save_model=True,
                        save_jpg_addr="save_file/Department_pic/Longevity_Pay/")

    plot_bar_with_n(department_describe_list, ["mean","std","min","50%"], list(label_and_count_set.keys()),
                    save_addr="save_file/Department_pic/Base_Salary/")
    plot_bar_with_n(department_describe_list, ["mean", "std", "min", "50%"], list(label_and_count_set.keys()),
                    save_addr="save_file/Department_pic/Overtime_Pay/",select_what_salary="Overtime_Pay")
    plot_bar_with_n(department_describe_list, ["mean", "std", "min", "50%"], list(label_and_count_set.keys()),
                    save_addr="save_file/Department_pic/Longevity_Pay/", select_what_salary="Longevity_Pay")
    return


if __name__ == '__main__':
    data_ori = load_and_init_dataSet()
    department_data_analysis(data_ori,78)
