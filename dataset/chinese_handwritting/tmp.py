from dataset.chinese_handwritting.hwdb import HWDB
from dataset.chinese_handwritting.olhwdb import OLHWDB


def run_match():
    online = OLHWDB.decode_pot_file('E:/Datasets/OLHWD/Pot1.0Test/121.pot')
    offline = HWDB.decode_gnt_file('E:/Datasets/HWD/Gnt1.0Test/121-t.gnt')

    for i in range(-5, -1):
        online[i].show()
        offline[i].show()


if __name__ == '__main__':
    run_match()
