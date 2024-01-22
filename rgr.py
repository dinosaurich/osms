import matplotlib.pyplot as plt
import numpy as np


def crc_gen(bit_arr):

    G = [1, 1, 1, 1, 0, 0, 0, 0]
    # Добавляем 7 нулей в конец
    bit_arr = np.concatenate((bit_arr, np.zeros(7, dtype=int)))
    
    for i in range(len(bit_arr)-8):
        if bit_arr[i] == 1:
            bit_arr[i:i+8] ^= G
    
    return bit_arr[-7:]

def crc_rec(bit_arr):

    G = [1, 1, 1, 1, 0, 0, 0, 0]  # Порождающий полином G
    bit_arr = np.array(bit_arr, dtype=int)
    for i in range(len(bit_arr)-8):
        if bit_arr[i] == 1:
            bit_arr[i:i+8] ^= G
    
    return bit_arr[-7:]

def gold_gen(G):

    x = [1, 0, 1, 1, 0]
    y = [1, 1, 1, 0, 1]
    gold_sequence = []
    
    for i in range(G):
        gold_sequence.append(x[4] ^ y[4])

        temp = x[3] ^ x[4]
        x = [temp] + x[:4]

        temp = y[1] ^ y[4]
        y = [temp] + y[:4]

    return np.array(gold_sequence)

def cor_reс(bit_arr, gold):

    gold = np.repeat(gold, 5)
    correlation = np.correlate(bit_arr, gold, "valid")
    max_arg = np.argmax(correlation)
    return max_arg

def decrypt(bit_arr):

    decod = []
    for i in range(0, len(bit_arr), 5):
        sr_arr = np.mean(bit_arr[i:i+5])
        if sr_arr > 0.5:
            decod.append(1)
        else:
            decod.append(0)
    return decod


f_name = input("Имя: ")
l_name = input("Фамилию: ")

bit_sequence = []
for char in f_name + ' ' +  l_name:
    ascii_code = ord(char)  
    binary_representation = format(ascii_code, '08b') 
    bit_sequence.extend([int(bit) for bit in binary_representation])

plt.figure(1)
plt.title('Битовая последовательность')
plt.plot(bit_sequence)
bit_string = ''.join(map(str, bit_sequence))
print('Бит. послед.',bit_string)
crc = crc_gen(bit_sequence)
print('CRC', crc)
gold_arr = gold_gen(31)
print('gold_arr', gold_arr)
data = np.concatenate((gold_arr, bit_sequence, crc))
plt.figure(2)
plt.title('Данные + CRC + синхронизации')
plt.plot(data)
data_x = np.repeat(data, 5) # 5 отчётов на 1 бит
plt.figure(3)
plt.title('Данные + CRC + синхронизации(амплитудная модуляция)')
plt.plot(data_x)
Nx_x2 = np.zeros(len(data_x)*2)
len_data = len(data_x)
print("Введите число начала передачи сигнала ( 0 -", len_data,")",end='')
start_sig = input()
start_sig = int(start_sig)
Nx_x2[start_sig:start_sig + len(data_x)] = data_x
plt.figure(4)
plt.title('Полученный сигнал')
plt.plot(Nx_x2)


noise = np.random.normal(0, 0.1, len(Nx_x2))
sig_noise = noise + Nx_x2
sig_noise2 = noise + Nx_x2
plt.figure(5)
plt.title('Полученный сигнал + шум')
plt.plot(sig_noise)



cor = cor_reс(sig_noise, gold_arr)
print(cor)
sig_noise = sig_noise[cor:]
plt.figure(6)
plt.title('Полученный сигнал начиная с синхронизации')
plt.plot(sig_noise)



print(len(data_x))
sig_noise = sig_noise[:len(data_x)]

dec = decrypt(sig_noise)
print('Полученный данные', dec)
plt.figure(7)
plt.title('Полученный сигнал начиная с синхронизации')
plt.plot(sig_noise)



dec = dec[len(gold_arr):]
print('Полученный данные без синхронизации', dec)


crc_sig = crc_rec(dec)
print('Проверка crc',crc_sig)
err = 0
for i in range(len(crc_sig)):
    if crc_sig[i] == 1:
        err = 1
if err == 0:
    print('Предача прошла без ошибок')
else:
    print('БЫЛИ ошибки!!')


dec = dec[:-7]

bit_string = ''.join(str(b) for b in dec)
"""n = int(bit_string, 2)
decoded_string = n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

print(decoded_string)"""

fft_tx = np.fft.fft(Nx_x2) + 30
fft_rx = np.fft.fft(sig_noise2)
plt.figure(8)
plt.title('Спектры полученного и переданного сигнала')
plt.plot(fft_tx) ## Без шума
plt.plot(fft_rx) ## С шумом


data5x = np.repeat(data, 5)
data10x = np.repeat(data, 10)
data20x = np.repeat(data, 20)

data10x = data10x[:len(data5x)]
data20x = data20x[:len(data5x)]

data5x = np.fft.fftshift(np.fft.fft(data5x))+80
data10x = np.fft.fftshift(np.fft.fft(data10x))+40
data20x = np.fft.fftshift(np.fft.fft(data20x))

plt.figure(9)
plt.title('Спектры сигналов')
plt.xlabel("Частота, Гц")
plt.ylabel("Амплитуда")
plt.plot(data5x)
plt.plot(data10x)
plt.plot(data20x)
plt.show()
