import numpy as np
import string
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
data = "\nTo be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles\nAnd by opposing end them. To die—to sleep,\nNo more; and by a sleep to say we end\nThe heart-ache and the thousand natural shocks\nThat flesh is heir to: 'tis a consummation\nDevoutly to be wish'd. To die, to sleep;\nTo sleep, perchance to dream—ay, there's the rub:\nFor in that sleep of death what dreams may come,\nWhen we have shuffled off this mortal coil,\nMust give us pause—there's the respect\nThat makes calamity of so long life.\nFor who would bear the whips and scorns of time,\nTh'oppressor's wrong, the proud man's contumely,\nThe pangs of dispriz'd love, the law's delay,\nThe insolence of office, and the spurns\nThat patient merit of th'unworthy takes,\nWhen he himself might his quietus make\n"
wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=200, max_font_size=40, random_state=42).generate(data)
(fig, axes) = plt.subplots(nrows=1, ncols=3, figsize=(24, 24))
axes[0].imshow(wordcloud)
axes[0].axis('off')
axes[1].imshow(wordcloud)
axes[1].axis('off')
axes[2].imshow(wordcloud)
axes[2].axis('off')
fig.tight_layout()
data = "\nTo be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles\nAnd by opposing end them. To die—to sleep,\nNo more; and by a sleep to say we end\nThe heart-ache and the thousand natural shocks\nThat flesh is heir to: 'tis a consummation\nDevoutly to be wish'd. To die, to sleep;\nTo sleep, perchance to dream—ay, there's the rub:\nFor in that sleep of death what dreams may come,\nWhen we have shuffled off this mortal coil,\nMust give us pause—there's the respect\nThat makes calamity of so long life.\nFor who would bear the whips and scorns of time,\nTh'oppressor's wrong, the proud man's contumely,\nThe pangs of dispriz'd love, the law's delay,\nThe insolence of office, and the spurns\nThat patient merit of th'unworthy takes,\nWhen he himself might his quietus make\n"
data = data.replace('\n', ' ')
data = data.lower()
data = data.translate(str.maketrans('', '', string.punctuation))
data = data[1:-1]
print(data)
chars = sorted(set(data))
char_to_idx = {c: i for (i, c) in enumerate(chars)}
idx_to_char = {i: c for (i, c) in enumerate(chars)}
(data_size, char_size) = (len(data), len(chars))
hidden_size = 10
weight_sd = 0.1
z_size = hidden_size + char_size
t_steps = 25

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y * y

def forward(x, u, q):
    z = np.row_stack((q, x))
    a = sigmoid(np.dot(wa, z) + ba)
    b = sigmoid(np.dot(wb, z) + bb)
    c = tanh(np.dot(wc, z) + bc)
    d = sigmoid(np.dot(wd, z) + bd)
    e = a * u + b * c
    h = d * tanh(e)
    v = np.dot(wv, h) + bv
    y = np.exp(v) / np.sum(np.exp(v))
    return (z, a, b, c, d, e, h, v, y)

def optimize(grads, theta, lr=0.05):
    (dwa, dwb, dwc, dwd, dwv, dba, dbb, dbc, dbd, dbv) = grads
    (wa, wb, wc, wd, wv, ba, bb, bc, bd, bv) = theta
    wa -= dwa * lr
    wb -= dwb * lr
    wc -= dwc * lr
    wd -= dwd * lr
    wv -= dwv * lr
    ba -= dba * lr
    bb -= dbb * lr
    bc -= dbc * lr
    bd -= dbd * lr
    bv -= dbv * lr
    return (wa, wb, wc, wd, wv, ba, bb, bc, bd, bv)
losses = {}
(z, a, b, c, d, e, h, v, y) = ({}, {}, {}, {}, {}, {}, {}, {}, {})
(q, x, u) = ({}, {}, {})
(wa, wb, wc, wd) = [np.random.randn(hidden_size, z_size) * weight_sd + 0.5 for x in range(4)]
(ba, bb, bc, bd) = [np.zeros((hidden_size, 1)) for x in range(4)]
wv = np.random.randn(char_size, hidden_size) * weight_sd
bv = np.zeros((char_size, 1))
q[-1] = np.zeros((hidden_size, 1))
u[-1] = np.zeros((hidden_size, 1))
pointer = 25
t_steps = 25
inputs = [char_to_idx[ch] for ch in data[pointer:pointer + t_steps]]
targets = [char_to_idx[ch] for ch in data[pointer + 1:pointer + t_steps + 1]]
for epoch in range(1000):
    loss = 0
    for t in range(len(inputs)):
        x[t] = np.zeros((char_size, 1))
        x[t][inputs[t]] = 1
        (z[t], a[t], b[t], c[t], d[t], e[t], h[t], v[t], y[t]) = forward(x[t], u[t - 1], q[t - 1])
        (u[t], q[t]) = (e[t], h[t])
        loss += -np.log(y[t][targets[t], 0])
    dh_next = np.zeros_like(q[0])
    de_next = np.zeros_like(u[0])
    (dwa, dwb, dwc, dwd, dwv, dba, dbb, dbc, dbd, dbv) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for t in reversed(range(len(inputs))):
        target = targets[t]
        dv = np.copy(y[t])
        dv[target] -= 1
        dwv += np.dot(dv, h[t].T)
        dbv += dv
        dh = np.dot(wv.T, dv)
        dh += dh_next
        dd = dh * tanh(e[t])
        dd = dsigmoid(d[t]) * dd
        dwd += np.dot(dd, z[t].T)
        dbd += dd
        de = np.copy(de_next)
        de += dh * d[t] * dtanh(tanh(e[t]))
        dc = de * b[t]
        dc = dtanh(c[t]) * dc
        dwc += np.dot(dc, z[t].T)
        dbc += dc
        db = de * dc
        db = dsigmoid(b[t]) * db
        dwb += np.dot(db, z[t].T)
        dbb += db
        da = de * u[t - 1]
        da = dsigmoid(a[t]) * da
        dwa += np.dot(da, z[t].T)
        dba += da
        dz = np.dot(wa.T, da) + np.dot(wb.T, db) + np.dot(wc.T, dc) + np.dot(dd.T, dd)
        dh_next = dz[:hidden_size, :]
        de_next = a[t] * de
    grads = (dwa, dwb, dwc, dwd, dwv, dba, dbb, dbc, dbd, dbv)
    theta = (wa, wb, wc, wd, wv, ba, bb, bc, bd, bv)
    (wa, wb, wc, wd, wv, ba, bb, bc, bd, bv) = optimize(grads, theta)
    losses[epoch] = loss
plt.plot(list(losses.keys()), [losses[x] for x in list(losses.keys())])