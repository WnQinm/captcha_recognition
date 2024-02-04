# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
from captcha_cnn_model import CNN
import base64
from PIL import Image
from io import BytesIO
from my_dataset import transform


def captchaRecog(img_base64, batch=False):
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('captcha_recog_model_weight.pt'))

    _, context=img_base64.split(",")
    img_data = base64.b64decode(context)
    image = Image.open(BytesIO(img_data)).convert("RGBA")

    W, L = image.size
    for h in range(W):
        for i in range(L):
            if image.getpixel((h, i))[-1] == 0:
                image.putpixel((h, i), (255, 255, 255, 255))
    image = transform(image.convert("RGB")).unsqueeze(dim=0)

    vimage = Variable(image)
    predict_label = cnn(vimage)

    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c = '%s%s%s%s' % (c0, c1, c2, c3)
    return c


base64_code = "data:image/png;base64,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"
print(captchaRecog(base64_code))