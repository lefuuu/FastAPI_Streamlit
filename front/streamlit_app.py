import streamlit as st 
import requests


st.title('Че то про FastAPI')
st.write('Че то вроде нужно сделать, модельку какую то подгрузить и она как будто бы должна работать, ща будем разбираться')
st.divider()
st.subheader('Туловище')
input = st.text_input('Ну попробуй сюда написать отзыв какой нибудь, возможно выведется тебе какой это отзыв, положительный или отрицательный, мб даже вероятность определения будет')
click = st.button('Аналазируй, падла')
if click:
    text = {'text' : input}
    res = requests.post('http://127.0.0.1:8000/clf_text', json=text)
    cls = res.json()['label']
    if cls == 'positive':
        cls = 'вообще добренький такой'
    elif cls =='neutral':
        cls = 'ну такой себе, какой то вялый, не рыба не мясо чисто нейтралитет сохраняет'
    else:
        cls = 'погань черная, хейтеры писали'
    acc = res.json()['prob']
    st.subheader('Аутпут')
    st.write(f'Походу отзыв {cls}. На {round(acc,2) * 100}% уверен, отвечаю')