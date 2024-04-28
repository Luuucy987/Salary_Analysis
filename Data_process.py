import pandas as pd
import os.path as osp
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_with_dataframe(data:pd.DataFrame,list_info:list ,model = 'default'):

    if list_info == None:
        print("[error] info list is empty")
        return

    if model == 'default':
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        sns.lineplot(data, ax=axs[0])
        sns.boxplot(data, ax=axs[1])
        sns.barplot(data, errorbar=None, ax=axs[2])
        sns.pointplot(data, ax=axs[3])
        # axs = set_plot_label_and_title(axs, list_info, ['_line', '_box', '_bar', 'point'])
        plt.tight_layout()
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
def plot_one_label_histogram(data_ori:pd.DataFrame, label:str, save_pic_addr:str = None ,save_model = False):
    """
    对上述statistics_sorted_data的一键傻瓜化，之间调用api即可实现->获取单个标签在数据集中的直方图
    步骤如下：1.获取单个标签信息 2.排序【从小到大】 3.调用statistics_sorted_data返回分布信息字典 4.对图像进行后处理操作
    :param data_ori:输入数据集
    :param label: 标签信息【只能是单个标签】
    :param save_pic_addr: 保存文件路径【非默认】
    :param save_model: 保存模式，默认是false，如果是true且路径ok就保存
    """
    plt.figure(figsize=(12, 6))
    temp = data_ori[[label]]
    print(f"[info]ready sort by {label} value")
    sort_temp = temp.sort_values(by=label)
    print(f"[info] building {label} histogram ")
    statis_data = statistics_sorted_data(sort_temp, label)

    statis_data = pd.DataFrame(statis_data, index=[0])
    print(f"[info]success sort by {label} value")

    sns.barplot(statis_data)
    plt.xticks(rotation=300)
    plt.tight_layout()
    plt.title(label+"label histogram")
    plt.xlabel("salary")
    plt.ylabel("count of "+label)
    if save_model == True and save_pic_addr:
        plt.savefig(save_pic_addr,bbox_inches='tight')
        print(f"[info] save {label} histogram in {save_pic_addr}")
        return
    else:
        plt.show()
        print(f"[info] build {label} histogram success")
        return

if __name__ == '__main__':
    fig, ax = plt.subplots(3, 2)

    M_select_dic = {'Gender': [init_analysis.MODEL_CAL_EQUAL, 'M'] }
    F_select_dic = {'Gender': [init_analysis.MODEL_CAL_EQUAL, 'F'] }
    data = load_and_init_dataSet()

    data_M = select_some_label(data, M_select_dic)
    data_select_M = cal_describe(data_M, init_analysis.CAL_DEFAULT_LIST)#对sex=男数据预处理

    sns.barplot(data_select_M['Base_Salary'], ax=ax[0,0], color="#00BFFF")
    sns.barplot(data_select_M['Overtime_Pay'], ax=ax[1,0], color="#FFFF00")
    sns.barplot(data_select_M["Longevity_Pay"], ax=ax[2,0], color="#00FF00")

    for i in range(2):
        ax[i, 0].set_ylabel("Salary")
    ax[0, 0].set_xlabel("Male Base_Salary")
    ax[1, 0].set_xlabel("Male Overtime_Pay")
    ax[2, 0].set_xlabel("Male Longevity_Pay")


    data_F = select_some_label(data, F_select_dic)
    data_select_F = cal_describe(data_F, init_analysis.CAL_DEFAULT_LIST)
    sns.barplot(data_select_F['Base_Salary'], ax=ax[0, 1], color="#00BFFF")
    sns.barplot(data_select_F['Overtime_Pay'], ax=ax[1, 1], color="#FFFF00")
    sns.barplot(data_select_F["Longevity_Pay"], ax=ax[2, 1], color="#00FF00")
    for i in range(3):
        ax[i, 1].set_ylabel("Salary")
    ax[1, 1].set_ylim(0, 20000)
    ax[2, 1].set_ylim(0, 4000)
    ax[0, 1].set_xlabel("Female Base_Salary")
    ax[1, 1].set_xlabel("Female Overtime_Pay")
    ax[2, 1].set_xlabel("Female Longevity_Pay")
    plt.tight_layout()
    plt.show()

    plot_one_label_histogram(data_F, "Base_Salary", "save_file/output_F.jpg", save_model=True)
    plot_one_label_histogram(data_M, "Base_Salary")



    # save_dataFarme(data_select_M, "save_file/end_M.xlsx")
    # save_dataFarme(data_select_F, "save_file/end_F.xlsx")
    print("")
