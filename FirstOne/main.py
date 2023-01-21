import random, math
import numpy as np


# DANE:
# wagi w0 – w5: double
# bias b1 - b3: double
# FUNKCJE:
# sigmoid(x: double) => 1 / (1 + exp(-x))
# derivsigmoid(x: double) => sigmoid(x) * (1 - sigmoid(x))
# mse_loss(y, ypred: array) => sr_arytm((y – ypred)^2)
# feedforward(x: array) => wynik neuronu
#
# train(data: array, y: array, learn_rate: double, epochs: double) => nauczy siec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    # if x < 0:
    #     return 1 - 1 / (1 + math.exp(x))
    # else:
    #     return 1 / (1 + math.exp(-x))


def derivsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def mse_loss(y, ypred):
    return (y - ypred)**2


class Neuron:
    def __init__(self):
        self.w0 = random.random()
        self.w1 = random.random()
        self.w2 = random.random()
        self.w3 = random.random()
        self.w4 = random.random()
        self.w5 = random.random()

        self.b0 = random.random()
        self.b1 = random.random()
        self.b2 = random.random()

    def feedforward(self, x: list):
        h0 = sigmoid(self.w0*x[0] + self.w1*x[1] + self.b0)
        h1 = sigmoid(self.w2*x[0] + self.w3*x[1] + self.b1)
        o0 = sigmoid(h0*self.w4 + h1*self.w5 + self.b2)
        return o0

    def train(self, data: list, correct: list, learn_rate, epochs: int, *, DEBUG=False, LOG=True):
        for epoch in range(epochs):
            for i in range(len(data)):
                x, y, expected = data[i][0], data[i][1], correct[i]

                h0temp = self.w0*x + self.w1*y + self.b0
                h0 = sigmoid(h0temp)

                h1temp = self.w2*x + self.w3*y + self.b1
                h1 = sigmoid(h1temp)

                o0temp = h0*self.w4 + h1*self.w5 + self.b2
                o0 = sigmoid(o0temp)

                ypred = self.feedforward([x, y])
                if ypred != o0:
                    print(False)

                # 1
                dMSE = -2 * (expected - ypred)

                # 2
                dB2 = derivsigmoid(o0temp)
                dW4 = h0 * derivsigmoid(o0temp)
                dW5 = h1 * derivsigmoid(o0temp)

                # 3
                dH0 = self.w4 * derivsigmoid(o0temp)
                dH1 = self.w5 * derivsigmoid(o0temp)

                # 4
                dB0 = derivsigmoid(h0temp)
                dW1 = y * derivsigmoid(h0temp)
                dW0 = x * derivsigmoid(h0temp)

                # 5
                dB1 = derivsigmoid(h1temp)
                dW3 = y * derivsigmoid(h1temp)
                dW2 = x * derivsigmoid(h1temp)

                # 6 aktualizacja
                self.w0 -= learn_rate * dW0 * dH0 * dMSE
                self.w1 -= learn_rate * dW1 * dH0 * dMSE
                self.w2 -= learn_rate * dW2 * dH1 * dMSE
                self.w3 -= learn_rate * dW3 * dH1 * dMSE
                self.w4 -= learn_rate * dW4 * dMSE
                self.w5 -= learn_rate * dW5 * dMSE

                self.b0 -= learn_rate * dB0 * dH0 * dMSE
                self.b1 -= learn_rate * dB1 * dH1 * dMSE
                self.b2 -= learn_rate * dB2 * dMSE
            if epoch % 100 == 0 and DEBUG:
                print(epoch)
                for i in range(len(data)):
                    ypred = self.feedforward([data[i][0], data[i][1]])
                    loss = mse_loss(correct[i], ypred)
                    print(data[i], correct[i], f'{loss}{(25-len(str(loss))) * " "}{ypred}')
                print()
        log = ""
        if LOG:
            for i in range(len(data)):
                ypred = self.feedforward([data[i][0], data[i][1]])
                loss = mse_loss(correct[i], ypred)
                log += f'{data[i]} {correct[i]} {loss}{(25-len(str(loss))) * " "}{ypred}\n'
            log += '\n'
        return log

# ==TRAIN==
# FOR 0..EPOCHS
#
# FOR X, Y IN DATA:
#
# INICJACJA NEURONU + LOSOWANIE_WAG()
# FEED_FORWARD() DLA CAŁEJ SIECI
# CALCULATE_MSE()
# CALCULATE DELTA (W4, W5, B2, H0, H1, W0, W1, B1, W2, W3, B2)
# AKUALIZACJA WAG
#
# IF EPOCHS % N == 0
#
# YPRED = FEED_FORWARD()
# MSE_LOSS(Y, YPRED)
# PRIN MSE AND YPRED


data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

correctAND = [
    0,
    0,
    0,
    1,
]

correctOR = [
    0,
    1,
    1,
    1,
]

correctXOR = [
    0,
    1,
    1,
    0,
]

correctXNOR = [
    1,
    0,
    0,
    1,
]

correctXAND = [
    1,
    0,
    0,
    1,
]

correctNAND = [
    1,
    1,
    1,
    0,
]

learn_rate = 0.1
epochs = 100000

N = Neuron()
print("AND")
log1 = N.train(data, correctAND, learn_rate, epochs)
print(log1)

print("OR")
log2 = N.train(data, correctOR, learn_rate, epochs)
print(log2)

print("XOR")
log3 = N.train(data, correctXOR, learn_rate, epochs)
print(log3)

print("XAND")
log4 = N.train(data, correctXAND, learn_rate, epochs)
print(log4)

print("XNOR")
log5 = N.train(data, correctXNOR, learn_rate, epochs)
print(log5)

print("NAND")
log6 = N.train(data, correctNAND, learn_rate, epochs)
print(log6)
