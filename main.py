

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

print(df.head())
print("############################################################################")
print("df.describe().T")
print(df.describe().T)
print("############################################################################")
print("df.isnull().sum()")
print(df.isnull().sum())
print("############################################################################")
print("df.shape")
print(df.shape)
print("############################################################################")

######### Görev 1:Veri Ön İşleme İşlemlerini Gerçekleştiriniz

######### 2010-2011 verilerilerini seçiniz ve tüm veriyi ön işlemeden geçiriniz.

######### Germany seçimi sonraki basamakta olacaktır

print("#################### GÖREV 1 ####################")
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True) #eksik değerleri kaldır, bu işlemi kalıcı bir şekilde yap
    dataframe = dataframe[~dataframe["Invoice"].astype(str).str.contains("C", na=False)] #iade işlemini ifade etmekte C ile başyalan Invoice'lar
    dataframe = dataframe[~dataframe["StockCode"].astype(str).str.contains("POST", na=False)] #iade işlemini ifade etmekte C ile başyalan Invoice'lar
    dataframe = dataframe[dataframe["Quantity"] > 0] #quantity negatif olamaz
    dataframe = dataframe[dataframe["Price"] > 0] #price negatif olamaz
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)


print("Veri Ön İşleme Sonrası Dataframe")
print(df.describe().T)
############################################
#ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)  #satırlarda invoicelar, sütunlarda productlar olsun istiyoruz ve bir faturaa bir ürünün olup olmaması 1 ve 0larla ifade edilsin istiyoruz.
############################################
#aşağıda yorum satırında erişmek istediğimiz tablo yapısı bulunmaktadır. invoice'lara sepet -miş gibi davranacağız

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

####Görev 2:
#Germany müşterileri üzerinden birliktelik kuralları üretiniz.
print("#################### GÖREV 2 ####################")

df_g = df[df['Country'] == "Germany"] #Almanya müşterilerinin birliktelik kurallarını türetmiş olacağız

print(df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20))
print(df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]) #her bir faturadaki her bir üründen kaç tane
#olduğu bilgisini aldım unstack() diyerek bunu pivot ediyorum. Yani description sütunundaki isimlendirmeleri değişken isimlendirmelerine çeviriyorum.
#iloc ile index based seçim yap satırlardan ve sütunlardan beşer tane getir diyorum.

#Boş olan yerlere nan eğer bir satın alma varsa da oraya değer geldi.
print(df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]) #eksik değerlerde 0, dolularda 1 yazsın
#24 yazan yerde 1 diğer yerlerde 0 yazması lazım

print(df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5])# değer 0'dan büyükse 1 yaz değilse 0 yaz, 5 satır 5 sütun göster
print(df_g.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5])

#apply fonksiyonu: satır ya da sütun bilgisi verilir, bir fonksiyonu satır ya da sütunlarda uygular, döngü yazılmasına gerek kalmadan.
#applymap fonksiyonu: tüm gözlemleri gezer

#yukarıdaki iki satır için fonksiyon yazılması gerekmektedir. Description veya StockCode verilebilmeli fonksiyona argüman olarak almasına bağlı olarak

#alttaki fonksiyon bu amaçla yazılmıştır.
def create_invoice_product_df(dataframe, id=False): #invoice product matrixini oluşturur, istersek stockcodelara göre istersek descriptionlara göre getirir
    if id: #id True ise işlemi StockCode'a göre yapıyor
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
    else: #id False ise işlemi Description'a göre yapıyor
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)

g_inv_pro_df = create_invoice_product_df(df_g) #id argümanı girilmediği durumda ön tanımlı olduğu için id değerini False alıyor

g_inv_pro_df = create_invoice_product_df(df_g, id=True)


def check_id(dataframe, stock_code):
    # girilen dataframe'den stockcode değişkeni seçilecek ve sorgulanması istenen stock_code un descriptionu gelecek. Çıktının içerisindeki değerlerden
    # sdece string değere erişmek istediğimizden dolayı values[0] diyoruz ve bunu bir listeye çeviriyoruz. ve product name i print ediyoruz
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


####Görev 3:
#ID'leri verilen ürünlerin isimleri nelerdir?
print("#################### GÖREV 3 ####################")

print("Kullanıcı 1 ürün id'si:")
check_id(df_g, 21987)
print("Kullanıcı 2 ürün id'si:")
check_id(df_g, 23235)
print("Kullanıcı 3 ürün id'si:")
check_id(df_g, 22747)
############################################
# Birliktelik Kurallarının Çıkarılması
############################################

#Öncelikle, Apriori fonksiyonu ile olası tüm ürün birlikteliklerinin support değerlerini yani olasılıklarını bulmak olmalıdır.
print(g_inv_pro_df)
#min_support'un diğer adı threshold, eşik değer.    use_colnames--> kullanılan veri setindeki değişkenlerin isimleri kullanılmak isteniyorsa colnames true denilir.
frequent_itemsets = apriori(g_inv_pro_df, min_support = 0.01, use_colnames = True) #olası ürün birlikteliklerinin olasılığı elde edilmiş olur fonksiyonun çıktısında
frequent_itemsets.sort_values("support", ascending = False)#supporta göre değerleri büyükten küçüğe doğru sıraladık. Çıktıda her bir ürünün olasılığını görürüz,
#bu veriyi kullanarak birliktelik kurallarını çıkaracağız.
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01) #association rule tablosunu getirir

print("######################### RULES #########################")
print(rules)

#antecedents --> önceki ürün demek
#consequents --> ikinci ürün demek
#antecedent support --> ilk ürünün tek başına gözlenme olasılığı, support değeri
#consequents support --> ikinci ürünün tek başına gözlenme olasılığı
#support --> antecendents ve consequents'in birlikte gözlenme olasılığı (X ürünü ile Y ürününün birlikte alınma olasılığı)
#confidence --> x ürünü alındığında ynin alınma olasılığı
#lift --> x ürünü satın alındığında y ürününün satın alınması olasılığı x kat artar
#leverage --> kaldıraç etkisi demek, lifte benzer. supportu yüksek olan değerlere öncelik verme eğilimindedir.Bundan dolayı ufak bir yanlılığı vardır.
#lift değeri ise daha az sıklıkta olmasına rağmen bazı ilişkilei yakalayabilmekedir, dolayısıyla bizim için daha değerli bir metriktir, yansız bir metriktir.
#conviction -->  y ürünü olmadan x değerinin beklenen frekansıdır. ya da diğer taraftan x ürünü olmadan y değerinin beklenen frekansıdır.

#pratikte genelde şu kombinasyonlarla işlem yapılır; mesela support değeri şu değerin üzerinde olsun ve confidence değeri
#şu değerin üzerinde olsun, ve lift değeri de şu değerin üzerinde olsun gibi birkaç tane olası kombinasyon üzerinden
#değerlendirmeler yapılabilir.
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)] #birden fazla koşul olduğunda köşeli parantez kullanılması gerekmektedir
##check_id(df_g, 21086)
rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False) #confidence biraz daha güvenilir
#metrik olarak düşünülebilir yani en azından biri satın alındığında diğerinin satın alınma olasılığını ifade ediyor


df = retail_data_prep(df)
def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df) #birliktelik kurallarının türetilmesi



############################################
####Görev 4:  Sepetteki kullanıcılar için ürün önerisi yapınız.
############################################
# Örnek:
# Kullanıcı örnek ürün id: 22492
# Bir ürün sepete eklendiğinde, o ürün ile birlikte neleri önerilmesi gerektiği bir sql tablosunda tutuluyor olur,


product_id = 22492  # örnek olarak kullanıcı sepetine bu ürünü eklemiş olsun
check_id(df, product_id)  # ürünün adı ['MINI PAINT SET VINTAGE']

sorted_rules = rules.sort_values("lift", ascending=False)  # hesaplamış olduğumuz kuralları lift değerine göre sıralıyoruz, programatik kolaylık olması
# için sıralıyoruz. Benim için lift önemlidir diyorum. Bu kişisel tercihe kalmış bir durumdur. #antecedents bölümünde gezilir, girilen id yakalandığında, consequents sütununda aynı
# indexteki ürün önerilir.

recommendation_list = []  # birden fazla ürün önerisi yapılma ihtimali olduğu için liste oluşturuluyor

for i, product in enumerate(sorted_rules["antecedents"]):  # sorted_rules'da enumerate metodunu kullanıyorum.
    # enumerate fonksiyonu, antecedents sütunundaki tüm satırları gezer(gezecek olan product), antecedents sütununda gezerken aynı zamanda index bilgilerini de
    # istiyorum. index bilgilerini gezecek olan da i. enumerate fonk. görevi;; normalde bir döngü ile antecedentsdeki tüm bilgileri gezebilirdik ama antecedentsde bir koşul yakaladığımda
    # index bilgisine de erişmek istiyorum ki o index bilgisine göre consquents sütunundaki ürünü seçeceğim. yani sadece anecedentsde gezmek yeterli değil, eş zamanlı olarak
    # indexte de gezebilmeli.

    for j in list(product):  # product'ın temsil ettiği antecedents sütunundaki frozen set yapısı, antecedents sütununda işlem yapabilmek için listeye çevirmem gerekiyor ve bu
        # listenin içinde de gezinmem gerektiğinden for j in.. yapısı kullanılmıştır.

        if j == product_id:  # eğer aradığımız ürün id'sini listede bulursak bunu recommendation_list' eklemesini istiyoruz. ekleyeceğimiz şey ise;
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])  # antecedents sütununda aradığım id'yi bulduğum satırın indexinde consequents sütununa karsılık gelen ürünü ekler listeye.
            # sorted_rules.iloc[i] -->diyerek enumerate ile türettiğim indexe ulaştım. sorted_rules.iloc[i]["consequents"]) --> o indexteki consquents sütunundaki demiş oldum.
            # sorted_rules.iloc[i]["consequents"])[0] --> bunu bir değer olarak getirmesini istediğim için ilk yakaladğını getir dedim [0] yapıyı kullanarak, çünkü iki üç tane olabilir.
            #

recommendation_list[0:3]  # sorted_rules'u lifte göre sıraladığım için; benim için satışı en çok artıracak olduğunu varsaydığım değerleri seçebilmek için
# ancak antecedents sütununda birden fazla aradığım değerden olabilir, bu sebeple benim önereceğim ürünlere kısıtlama getirmem gereklidir.


############################################
####Görev 5:  Önerilen ürünlerin isimleri nelerdir?
print("#################### GÖREV 5 ####################")
check_id(df, 22328)  # öneri olarak gelen ürünlerin isimlerine bakabilmek için

for i in recommendation_list:
    check_id(df, i)

def arl_recommender(rules_df, product_id, rec_count=1): #yukarıdaki işlemlere fonksiyon yazılmıstır
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

arl_recommender(rules, 21987, 1)
arl_recommender(rules, 23235, 2)
arl_recommender(rules, 22747, 3)