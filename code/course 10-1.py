# 验证码生成库
from captcha.image import ImageCaptcha # pip install captcha
import numpy as np
from PIL import Image
import random
import sys

number = ['0','1','2','3','4','5','6','7','8','9']
# letter = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# LETTER = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
CAPTCHA_SAVE_DIR = "D:/pyy/tensorflow/code/captcha/images"

'''随机生成4个数字的字符串成。
char_set：用于生成的字符list
captcha_size：生成的验证码位数
'''


def random_captcha_text(char_set=number, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        # 随机选择
        c = random.choice(char_set)
        # 加入验证码列表
        captcha_text.append(c)
    return captcha_text


'''生成字符对应的验证码'''


def gen_captcha_text_and_iamge():
    image = ImageCaptcha()
    # 获得随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码列表转为字符串
    captcha_text = "".join(captcha_text)
    # 生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, 'D:/pyy/tensorflow/code/captcha/images'+captcha_text + ".jpg")  # 写到文件


# 循环生成10000次，但是重复的会被覆盖，所以<10000
num = 10000
if __name__ == "__main__":
    for i in range(num):
        gen_captcha_text_and_iamge()
        sys.stdout.write("\r>> Creating image %d/%d" % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()

    print("Generate finished.")