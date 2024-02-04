# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha
from PIL import Image
import random
import time
import captcha_setting
import os

def random_captcha():
    captcha_text = []
    for i in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)

# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=captcha_setting.IMAGE_WIDTH, 
                         height=captcha_setting.IMAGE_HEIGHT, 
                         fonts=["C:\\Windows\\Fonts\\msyhl.ttc", "C:\\Windows\\Fonts\\corbell.ttf", "C:\\Windows\\Fonts\\corbelli.ttf", "C:\\Windows\\Fonts\\segoeuil.ttf", "C:\\Windows\\Fonts\\seguili.ttf", "C:\\Windows\\Fonts\\STXIHEI.TTF"], 
                         font_sizes=(30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40)
                         )
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text)).convert('RGB')
    return captcha_text, captcha_image

if __name__ == '__main__':
    count = 1000
    path = captcha_setting.TEST_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        now = str(int(time.time()))
        text, image = gen_captcha_text_and_image()
        filename = text+'_'+now+'.png'
        image.save(path  + os.path.sep +  filename)
        print('saved %d : %s' % (i+1,filename))

