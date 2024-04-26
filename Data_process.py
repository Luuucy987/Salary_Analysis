import pandas as pd
import os.path as osp

class init_analysis():#默认信息类，用于默认初始化dataframe信息
    FILE_ADDR = "Employee_Salaries.csv"         #输入文件名
    OUTPUT_ADDR = "end.csv"                     #输出文件名
    LABEL_TUPLE = ['Department', 'Department_Name', 'Division', 'Gender', 'Base_Salary',
       'Overtime_Pay', 'Longevity_Pay', 'Grade']    #csv的行标签元素
    MODEL_CAL_MORE = 1  #指代运算符模式，1为大于、-1为小于、0为等于
    MODEL_CAL_LESS = -1
    MODEL_CAL_EQUAL = 0
    SELECT_DICT = {'Department': [MODEL_CAL_EQUAL, 'ABS']} #默认筛选样本的字典

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

    print(data_select.head())
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
    print("[info] data save process end")
    return 1
if __name__ == '__main__':
    test_select_dic = {'Department': [init_analysis.MODEL_CAL_EQUAL, 'ABS'], 'Gender': [init_analysis.MODEL_CAL_EQUAL, 'M'] }
    data = load_and_init_dataSet()
    # data_temp = data[data['Gender'] == 'M']
    # print(data_temp)
    data_select = select_some_label(data, test_select_dic)
    save_dataFarme(data_select, "sava_file/end.xlsx")
