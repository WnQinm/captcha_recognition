# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
from captcha_cnn_model import CNN
import base64
from PIL import Image
from io import BytesIO


def captchaRecog(img_base64):
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pt'))

    _, context=img_base64.split(",")
    img_data = base64.b64decode(context)
    image = Image.open(BytesIO(img_data))

    vimage = Variable(image)
    predict_label = cnn(vimage)

    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c = '%s%s%s%s' % (c0, c1, c2, c3)
    return c