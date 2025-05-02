# 车牌数据
numbers = list('0123456789')
alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
#            'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
#            'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
#            'zh_zang', 'zh_zhe']
chinese = [
    '川', '鄂', '赣', '赣', '贵', '贵', '黑', '沪', '冀', '津',
    '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼',
    '陕', '苏', '晋', '皖', '湘', '新', '豫', '豫', '粤', '云',
    '藏', '浙'
]
char_table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '川', '鄂', '赣', '甘', '贵',
              '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', '闽', '宁', '青', '琼', '陕', '苏', '晋',
              '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙']
# 车牌长宽
car_plate_w, car_plate_h = 136, 36

# 字符长宽
char_w, char_h = 20, 20


