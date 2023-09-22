# def preprocessing(text):
#     text =  re.sub(r'[\n+]', '', str(text))
#     text =  re.sub(r'[^\w\s]', '', text)
#     text =  re.sub(r'[^\x00-\x7F]+', '', text)
#     text =  re.sub(r'(pagi|siang|sore)', '', text, flags=re.IGNORECASE)
#     text =  re.sub(r'(dok|dokter)', '', text, flags=re.IGNORECASE)
#     text =  re.sub(r'(terimakasih|terima kasih|alo|alodokter)', '', text, flags=re.IGNORECASE)
#     text =  re.sub(r'\s+', ' ', text).strip().lower()
#     return text.strip().lower()


# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

# def tokenize(text):
#     tokens = word_tokenize(text)  # Tokenisasi teks menjadi kata-kata
#     stemmed_tokens = [stemmer.stem(token) for token in tqdm(tokens, desc="Processing tokens", leave=False)]
#     return stemmed_tokens



# start_time = time.time()

# token = tokenize(str(text))

# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Waktu yang diperlukan:", elapsed_time, "detik")




# df2 = pd.DataFrame()

# #menggunakan loop untuk setiap kategori unik dalam df
# for k in df.kategori.unique():
#     #mengambil subset dataframe berdasarkan kategori
#     sub = df[df.kategori == k]
#     #jika jumlah baris dalam subset lebih dari 6000
#     if len(sub) > 4000:
#         #mengambil 5000 baris secara acak dari subset dan menambahkannya ke df2 dengan pd.concat
#         sub = sub.sample(n=4000)
#         df2 = pd.concat([df2, sub])
#     #jika tidak, menambahkan seluruh subset ke df2 dengan pd.concat
#     else:
#         df2 = pd.concat([df2, sub])

# df2.kategori.value_counts()




# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)