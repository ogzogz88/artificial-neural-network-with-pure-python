#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np

# sigmoid fonksiyonunun hesaplanması
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# sigmoid fonksiyonunun türevinin hesaplanması
def sigmoid_derivative(x):
    return x * (1.0 - x)


# tam-bağlı(fully-connected) oluşturulan yapıyı istenen şekilde yerel-bağlı hale getirmek için,
# ağırlık değerlerini düzenleyen fonksiyon. Bağlı OLMAYAN noktalara karşılık gelen ağırlıklara
# 0 (sıfır) değerini atıyoruz.
'''
[[X X X 0 0 0]
 [0 X X X 0 0]
 [0 0 X X X 0]
 [0 0 0 X X X]]

'''
def partially_connect(w):
    len = np.shape(w)[1]
    i, j, k = 3, 4, 5
    for row in w:
        row[i], row[j], row[k] = 0, 0, 0
        i, j, k = (i+1)%len, (j+1)%len, (k+1)%len
            
    return w
    #print("inside partially connect")
    #print("w")
    #print(w)


'''
NeuralNetwork: yapay sinir ağı sınıfı

parametreler
------------
x: girdi değerleri (input values)
y: çıktı değerleri (output values)
'''
class NeuralNetwork:
    def __init__(self, x, y):
        
        # öğrenme oranı
        self.lr = 0.95
        
        # matris boyutu: (16*5)
        self.input      = x
        
        # matris boyutu:(5*7)
        # gizli katmanda (layer1) 6 adet node var, normalde birinci ağırlık(weights1) matrisi boyutunun (5*6) olması yeterli.
        # Ancak gizli katmana direkt olarak giren bias-2'yi kullanacağımız için gizli katman (16*7) boyutunda olmalı.
        # Bu sonucu elde edecek matris çarpımlarının yapılabilmesi ve aynı zamanda bias-1' ait ağırlık değerlerini tutabilmek
        # için birinci ağırlık değerleri matrisini (5*7) boyutuna getirdik. 
        self.weights1   = np.random.rand(self.input.shape[1],7) 
        
        # birinci ağırlık değerleri(weights1)'ne matris çarpımlarını yapabilmek için eklenen ekstra noktalara 
        # 0 (sıfır) değeri atıyoruz.
        # for idx,w in enumerate(self.weights1):
        #   self.weights1[idx][6] = 0
        
        # matris boyutu: (7*2)
        self.weights2   = np.random.rand(7,2)  
        
        # beklenen çıktı değeri
        # matris boyutu: (16*2)
        self.y          = y
        
        # tahmin edilen çıktı değeri
        # matris boyutu: (16*2)
        self.output     = np.zeros(self.y.shape)
        
        #print(self.weights1)
        #print(self.weights2)
        #print(self.output)
        
        # giriş katmanı ve gizli katman arasındaki ağırlık değerlerini yerel-bağlı hale getirdik.
        self.weights1 = partially_connect(self.weights1)
        print("self weights1")
        print(self.weights1)
        

    
    def calculate_RMSE(self):
        error = [(actual - predicted) for actual, predicted in zip(self.y, self.output)]
        square_error = [e**2 for e in error]
        mean_square_error = sum(square_error)/len(square_error)
        root_mean_square_error = mean_square_error**0.5
        
        return(root_mean_square_error)
    
    
    
    
        # feedforward akışı
        # input:(16*5) . weights1:(5*7) => layer1:(16*7)
        # layer1:(16*7) . weights2:(7*2) => output:(16*2)
        
        # backpropagation akışı
        # GD(Gradient Descent)derivative of loss-function: (2*(self.y - self.output) * sigmoid_derivative(self.output))
        # layer1.T:(6*16) . GD:(16*2) => d_weights2:(6*2)
        # input.T:(4*16) . ( GD:(16*2) . weights2.T:(2*6) * sigmoid_derivative(self.layer1) ) => d_weights1:(4*6)
    
    
    def feedforward(self):
        
        # np.dot(): 2 array için arrayin karşılıklı elemanlarını çarpar ve sonuç olarak bu çarpımların toplamını verir.
        # kaynak: https://nnfs.io/blq/
        # np.dot(): 1 array ve bir matrix çarpımında, matrix'in her bir dizisi(vektörü) için yukarıdaki çarpımı yapar,
        # çarpımları toplar ve sonuç olarak bu toplamları array içinde döndürür.
        # kaynak: https://nnfs.io/cyx/
        
        # gizli katmanı(layer1) hesaplarken giriş (input) ve birinci ağırlık(weights1) değerlerini noktasal/vektörel çarptık ve
        # sigmoid fonksiyonu ile aktivasyonu uyguladık.
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) )
        
        # enumerate ile "for in" döngüsü içerisinde indeks değerlerine erişebildik
        # gizli katmana ekstra eklediğimiz noktalar bias-2 değerlerini tutuyor. bias-2'ye de "1" değerini atadık.
        for idx, l_row in enumerate(self.layer1):
            self.layer1[idx][6] = 1
            
        #print("*-*-*--*-")
        #print(self.layer1)
        
        # çıkış katmanını(output layer) hesaplarken gizli katman(layer1) ve ikinci ağırlık(weights2) değerlerini 
        # noktasal/vektörel çarptık ve sigmoid fonksiyonu ile aktivasyonu uyguladık.
        self.output = sigmoid(np.dot(self.layer1, self.weights2) )
         
        #print("output => (16*6)*(6*2) = 16*2")
        #print(self.output)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #   7*2                7*16             16*2     16*2
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        
        #   5*7                5*16                      16*2     16*2                                            2*7
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # ağırlıkları kayıp fonksiyonunun (loss function) türevi ile güncelliyoruz.
        # birinci ağırlık değerlerini güncellerken de yerel-bağlı hale getirmeliyiz, çünkü güncelleme esnasında
        # yerel bağlılık bozuluyor.
        self.weights1 += partially_connect(d_weights1) * self.lr
        self.weights2 += d_weights2 * self.lr
    




# X(input): 16*5
# son satırdaki "1" değerleri bias
X = np.array([[0,0,0,0,1],
              [0,0,0,1,1],
              [0,0,1,0,1],
              [0,0,1,1,1],
              [0,1,0,0,1],
              [0,1,0,1,1],
              [0,1,1,0,1],
              [0,1,1,1,1],
              [1,0,0,0,1],
              [1,0,0,1,1],
              [1,0,1,0,1],
              [1,0,1,1,1],
              [1,1,0,0,1],
              [1,1,0,1,1],
              [1,1,1,0,1],
              [1,1,1,1,1],
             ])

# y(output): 16*2
y = np.array([[0,1],[0,1],[1,1],[1,0],[0,0],[0,1],[0,1],[1,0],[0,1],[0,1],[0,0],[1,1],[1,0],[0,0],[0,1],[1,0]])


nn = NeuralNetwork(X,y)


rmse = nn.calculate_RMSE()
print("rmse")
print(rmse)

iter = 1

while(rmse[0] > 0.01 or rmse[1] > 0.01):
    nn.feedforward()
    nn.backprop()
    rmse = nn.calculate_RMSE()
    print("**************")
    print("rms")
    print(rmse)
    print("iter")
    print(iter)
    iter+=1
    print("**************")


print("----------------------rmse----------------------")
print(rmse)
print("----------------------output----------------------")
print(nn.output)


# In[ ]:





# In[ ]:




