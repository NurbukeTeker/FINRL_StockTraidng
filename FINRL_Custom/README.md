FINRL Custom Dosyası, FIRNL orijinal kütüphanesindeki environment, model, ön işleme, veri çekme vb. birçok parçanın AYBANK kapsamında özelleştirilmiş halini kapsar.
Aşağıda genel olarak ve ayrıntılı oalcak şekilde tüm dosyalardan ve çalışma prensiplerinden bahsedilmiştir.

1- config.py
Config içerisinde environemnt ve agentlar için birçok ana belirleyici parametreler yer alır. Bir çalışma ve deney sırasında burada yapılan değişiklikler ile kodlar ve deneyler arasındaki bağımlı ve bağımsız değişkenleri değiştirebilir veya uygulayabiliriz.

2- customer.py 
Bu dosya real-time daki gibi, bir kullanıcı varmış gibi , belirli bir initial amount, başlangıç stockları ile belirlenebilen bir "customer" objesi oluşturu ve bu kullanıcı üzerinden RL işler.
Customer initialAmount ve customer initialstock değerleri burada oluşuturlur ve Env kısmındaki birçok parametre buradan çekilir.

3- yahoodownloader.py
FINRL kütüphanesinden alınan yahoodownload kullanılarak multi-stock verilerinin çekilmesini sağlar.

4- Preprocess.py ve process_data.py
Bu dosyalarda genel oalrak, stock datasının model eğitimine uygun hale getirilmesi ve teknik indikatörlerin eklenmesi kısmında kullanılır. Bu aşamalarda config.py'da belirlenen teknik indikaötler dataframe eklenir.

5-GetData.py 
Ana dosyalar üzerinden çalışıtırılırken verilerin çekildiği, işlendiği, prefill vb. işlemlerin yapıldığı, preprocess işelmlerinin çağrıldığı kısımdır. Burada yfinance veya yahoofinance kullanılarak configde belirlenne STAR,END dateleri baz alacak şekilde ve TICKER'ları alcak şekilde verileri çeker. Gerkeli işlemlerden sonra porcessed_data df içinde ana dosyaya geri döndürür.


6- Agent_Custom.py
Agent_custom dosyasında genel RL algortimalarının kullanıldığı agnet yapısının dinamik versiyonudur. FINRL'deki versiyonu direk olarka kullanılmaktadır. Burada DDPG, SAC, A2C, TD3 ve PPO oalracak şekilde model algortimaları seçilip, train ve trade(prediction) fonksiyonları çalışıtırlabilir ve Model-agent işler.

7- Environmentlar:
Environement_Custom.py : FINRL kütüphanesidnen alınan Env. 
forex_env_test.py:  FOREX env (LONG SHORT içerir)
ohlvc_FOREX.py:  FOREX env (LONG SHORT içerir)
EnvironementOhlvc.py : LSTM kullanılarak çalıştıırlan env içeirir

8-feature_extractor.py
LSTM için flatten katmanı yerine LSTM katmanının, LSTM env ile çalışıtırldığı ve modelde tanımlandığı kısmı kapsar. Model algoritması DDPG, SAC, A2C.. olurken, extractor kısmında LSTM entegre edilir.

9- Model Dosyaları: (main_A2C, main_DDPG, main_SAC, main_TD3, main_PPO)
Bu dosyaların her biri her bir algoritma ile verilen environment bazında RL süreçlerini çalıştırır. 
Bu aşamada verilerin getdata ile çekilmesi, train,trade oalrak ayrılması, Agent'ın algoritması ile create edilmesi, Train ve Trade environemnetlerının oluşturulması, Train environemntında agentın train edilmesi; trade environmentında predict(trade)işlemleri yapılır. Ve sonuçları check edilir.

10-Ensemble.py 
Environment olarak 3 adet algoritmanın kullanıldığı bir ensemble modeli içerir.


11- main.py 
Model algoritmalarının tek yerden çalıştırılması için konumlandıırlmıştır. Burada dire main_DDPG.py , main_PPO.py ... dosyaları çalıştırılır.

12- RL_DOW30.py
DOW30 Verilerinin çekildiği, modellerin ve environmetın algortima ismine göre dinamik bir şekilde oluşuturlduğu, eğitildiği, test edildiği ve sonuçlarının döndürüldüğü kısımları içerir.
Burada env olarak Environement_Custom.py ve EnvironementOhlvc.py  kullanılır.

13- RL_DOW30_LSTM.py
RL_DOW30.py'a aynı olcak şekilde ancak LSTM environmentını kullanır. Burada LSTM enviroenmentını DOW30 ile train-trade yaparız.

14- PlotGraph.py
Yapılan çalışmalardki görselleştirme kısmı için kullanılır. Graph çizer.

15- Backtesting.py
FINRL'den alınan ve trade sonuçlarının backtesting ile değerlendirildiği ksımı kapsar.
