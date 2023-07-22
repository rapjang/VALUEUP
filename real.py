
# 패키지 임포트
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime
import joblib
import pickle
import folium
import branca
import sklearn as sk
import ssl
import os
import requests
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from geopy.geocoders import Nominatim
from haversine import haversine
from urllib.request import urlopen
import xgboost
# 문자 보내는 패키지
# from twilio.rest import Client
import geopandas as gpd
from geopandas import GeoDataFrame

# ========================== 필요 함수 생성 ===============================  

# geocoding : 거리주소 -> 위도/경도 변환 함수
# Nominatim 파라미터 : user_agent = 'South Korea', timeout=None
# 리턴 변수(위도,경도) : lati, long
# 참고: https://m.blog.naver.com/rackhunson/222403071709
def geocoding(address):
    geolocoder = Nominatim(user_agent = 'South Korea', timeout=None)
    geo = geolocoder.geocode(address)

    latitude = geo.latitude
    longitude = geo.longitude
    return latitude, longitude


# preprocessing : '발열', '고혈압', '저혈압' 조건에 따른 질병 전처리 함수(미션3 참고)
# 리턴 변수(중증질환,증상) : X, Y
def preprocessing(desease):
    
    desease['발열'] = desease['체온'].apply(lambda x: 1 if x>= 37 else 0)
    desease['고혈압'] = desease['수축기 혈압'].apply(lambda x: 1 if x>= 140 else 0)
    desease['저혈압'] = desease['이완기 혈압'].apply(lambda x: 1 if x<= 90 else 0)

    target = '중증질환'
    Y = desease.loc[:, target]
    feature = ['체온', '수축기 혈압', '이완기 혈압', '호흡 곤란', 
               '간헐성 경련', '설사', '기침', '출혈', '통증', 
               '만지면 아프다', '무감각', '마비', '현기증', '졸도',
               '말이 어눌해졌다', '시력이 흐려짐', '발열', '고혈압', '저혈압']
    X = desease.loc[:, feature]
    
    return X, Y


# predict_disease : AI 모델 중증질환 예측 함수 (미션1 참고)
# 사전 저장된 모델 파일 필요('./xgb.pkl')
# preprocessing 함수 호출 필요 
# 리턴 변수(4대 중증 예측) : sym_list[pred_y_XGC[0]]
def predict_disease(patient_data):
    
    sym_list = ['뇌경색', '뇌출혈', '복부손상', '심근경색']
    test_df = pd.DataFrame(patient_data)
    test_x, test_y = preprocessing(test_df)
    model_XGC = joblib.load('./xgb.pkl')
    pred_y_XGC = model_XGC.predict(test_x)
    return sym_list[pred_y_XGC[0]]


# find_hospital : 실시간 병원 정보 API 데이터 가져오기 (미션1 참고)
# 리턴 변수(거리, 거리구분) : distance_df
def find_hospital(special_m, lati, long):
    
    # 미션1에서 저장한 병원정보 파일 불러오기 
    solution_df = pd.read_csv('./hospital_list.csv', encoding = 'cp949')

    ### 중증 질환 수용 가능한 병원 추출
    ### 미션1 상황에 따른 병원 데이터 추출하기 참고

    if special_m == "중증 아님":
        condition1 = (solution_df['응급실포화도'] != '불가')
        distance_df = solution_df[condition1].copy()
    else:
        condition1 = (solution_df[special_m] == 'Y') & (solution_df['가용수술실수'] >= 1)
        condition2 = (solution_df['응급실포화도'] != '불가')

        distance_df = solution_df[condition1 & condition2].copy()

    ### 환자 위치로부터의 거리 계산
    distance = []
    patient = (lati, long)
    
    for idx, row in distance_df.iterrows():
        distance.append(round(haversine((row['위도'], row['경도']), patient, unit='km'), 2))

    distance_df['거리'] = distance
    distance_df['거리구분'] = pd.cut(distance_df['거리'], bins=[-1, 2, 5, 10, 100],
                                 labels=['2km이내', '5km이내', '10km이내', '10km이상'])
            
    return distance_df


# 카카오모빌리티 최단경로 api 호출
def kakao_road(patient_point, destination_point):
    url = 'https://apis-navi.kakaomobility.com/v1/directions?'
    key = ''

    headers = {
        'Authorization': 'KakaoAK ' + os.getenv('REST_API_KEY', key),
    }

    params = {
        'origin' : patient_point,          # 환자위치 (예 : (127, 37)) 
        'destination' : destination_point  # 목적지 
    }

    response = requests.get(url, params=params, headers = headers)
    road_df = pd.DataFrame(response.json()['routes'][0]['sections'][0]['guides'])
    road = [[row['y'], row['x']] for idx, row in road_df.iterrows()]
    
    return road


def send_message(special_m, name, distance):
    
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=f"119 서비스 입니다 {special_m} 대응 위치는 {name} 이며 거리는 {distance}입니다.",
        from_='',
        to='')
    
    return message.sid


# -------------------- ▼ 필요 변수 생성 코딩 Start ▼ --------------------

data = pd.read_csv('./119_emergency_dispatch.csv', encoding="cp949")

## 오늘 날짜
now_date = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
now_date2 = datetime.datetime.strptime(now_date.strftime("%Y-%m-%d"), "%Y-%m-%d")

## 2023년 최소 날짜, 최대 날짜
first_date = pd.to_datetime("2023-01-01")
last_date = pd.to_datetime("2023-12-31")

## 출동 이력의 최소 날짜, 최대 날짜
min_date = datetime.datetime.strptime(data['출동일시'].min(), "%Y-%m-%d")
max_date = datetime.datetime.strptime(data['출동일시'].max(), "%Y-%m-%d")


# -----------------------전국 AED(자동심장충격기) 데이터 -----------------------

aed_df = pd.read_csv('전국자동심장충격기.csv', encoding = 'cp949')
tmp = aed_df.loc[aed_df['region'] == '광주광역시']

# ----------------------- session_state 초기화 ---------------------

# location : 환자의 위치
if "location" not in st.session_state:
    st.session_state["location"] = ""

# patient_la : 환자의 경도 (125)
if "patient_la" not in st.session_state:
    st.session_state["patient_la"] = ""
    
# patient_lo : 환자의 위도 (37)
if "patient_lo" not in st.session_state:
    st.session_state["patient_lo"] = ""
    
# special_m : 중증질환
if "special_m" not in st.session_state:
    st.session_state["special_m"] = ""
    
# real_tmp : 자동심장충격기 상위 5개 행 데이터프레임
if "real_tmp" not in st.session_state:
    st.session_state["real_tmp"] = ""

    
# =============== 통계를 위한 지도시각화 데이터 ====================
gwangju_grid = gpd.GeoDataFrame.from_file("./gwangju/emergency.shp", encoding='cp949')
gwangju_grid = gwangju_grid.to_crs("EPSG:4326")

# 각 중심점을 담는 리스트
dong_point = gwangju_grid.centroid

# -------------------- ▲ 필요 변수 생성 코딩 End ▲ --------------------

# ------------------------------------------------------------------------------------------------------------------------

st.set_page_config(layout='wide')

with st.sidebar:
    #st.sidebar.image('logo.png')
    tabs = option_menu('응급원격구조지시', ['응급환자정보', 'AED(심근경색)', '병원조회', '통계'], 
                       icons=['person', 'search', 'search', 'kanban'], 
                       menu_icon="plus", 
                       default_index=0,
                       styles={
        "container": {"padding": "0!important", "background-color": "#262728"},
        "icon": {"color": "white", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"10px", "--hover-color": "#333333"},
        #"nav-link-selected": {"background-color": "gray"}  # 사이드바 배경색 => 수정요망
        })


if tabs == '응급환자정보':

    # 환자정보 넣기
    st.markdown('#### 환자 정보')
    
    ## -------------------- ▼ 1-1그룹 날짜/시간 입력 cols 구성(출동일/날짜정보(input_date)/출동시간/시간정보(input_time)) ▼ --------------------
     
    col110, col111, col112, col113 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col110:
        st.info("출동일")
    with col111:
        input_date = st.date_input('출동 일자', label_visibility="collapsed")
    with col112:
        st.info("출동시간")
    with col113:
        input_time = st.time_input('출동 시간', datetime.time(now_date.hour, now_date.minute), 
                                   label_visibility="collapsed", step=60)

    ## -------------------------------------------------------------------------------------

    ## -------------------- ▼ 1-2그룹 
    ## 이름/성별 입력 cols 구성(이름/이름 텍스트 입력(name)/
    ## 나이/나이 숫자 입력(age)/
    ## 성별/성별 라디오(patient_s)) ▼ --------------------
    col120, col121, col122, col123, col124, col125 = st.columns([0.1, 0.3, 0.1, 0.1, 0.1, 0.1])
    with col120:
        st.info("이름")
    with col121:
        name = st.text_input('', label_visibility='collapsed')
    with col122:
        st.info("나이")
    with col123:
        age = number = st.number_input('', label_visibility='collapsed', format = '%d', step = 1)
    with col124:
        st.info("성별")
    with col125:
        patient_s = st.radio('',('남', '여'), label_visibility='collapsed')
        
    
    ##-------------------------------------------------------------------------------------
    ## -------------------- ▼ 1-3그룹 체온/환자위치(주소) 입력 cols 구성
    ## (체온/체온 숫자 입력(fever)
    ## /환자 위치/환자위치 텍스트 입력(location)) ▼ --------------------
    col130, col131, col132, col133 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col130:
        st.info("체온")
    with col131:
        fever = st.number_input('', label_visibility='collapsed',
                                min_value=30.00, max_value=60.00, step = 0.01, value = 36.5)
    with col132:
        st.info("환자 위치")
    with col133:
        st.session_state.location = st.text_input('', value = '광주광역시', label_visibility='collapsed', key = 'string')
        
    #### 거리주소 -> 위도/경도 변환 함수 호출
    st.session_state.patient_la, st.session_state.patient_lo = geocoding(st.session_state['location'])

    distance = []
    patient = (st.session_state.patient_la, st.session_state.patient_lo)
    
    # AED와의 거리 데이터프레임 설정
    for idx, row in tmp.iterrows():
        distance.append(int(haversine((row['wgs84Lat'], row['wgs84Lon']), patient, unit='m')))

    tmp['거리'] = distance
    tmp = tmp.sort_values(by = '거리', ascending = True)

    st.session_state["real_tmp"] = tmp.iloc[0:5, :]
    
    ## 확인용코드
    #st.dataframe(tmp)
    #st.dataframe(st.session_state.real_tmp)

    
    ##-------------------------------------------------------------------------------------

    ## ------------------ ▼ 1-4그룹 혈압 입력 cols 구성(
    # 수축기혈압/수축기 입력 슬라이더(high_blood)/
    # 이완기혈압/이완기 입력 슬라이더(low_blood)) ▼ --------------------
    ## st.slider 사용
    col140, col141, col142, col143 = st.columns([0.1, 0.3, 0.1, 0.3])
    with col140:
        st.info("수축기혈압")
    with col141:
        # 140이상 고혈압, 90이하 저혈압
        high_blood = st.slider('수축기혈압', min_value=0, max_value=300)
    with col142:
        st.info("이완기혈압")
        # 90이상 고혈압, 60이하 저혈압
    with col143:
        low_blood = st.slider('이완기혈압', min_value=0, max_value=200)
    

    ##-------------------------------------------------------------------------------------

    ## -------------------- ▼ 1-5그룹 환자 증상체크 입력 
    ## cols 구성(증상체크/checkbox1/checkbox2/checkbox3/checkbox4/checkbox5/checkbox6/checkbox7) ▼ -----------------------    
    ## st.checkbox 사용
    ## 입력 변수명1: {기침:cough_check, 간헐적 경련:convulsion_check, 
    #마비:paralysis_check, 무감각:insensitive_check, 
    #통증:pain_check, 만지면 아픔: touch_pain_check}
    ## 입력 변수명2: {설사:diarrhea_check, 출혈:bleeding_check, 
    ## 시력 저하:blurred_check, 호흡 곤란:breath_check, 현기증:dizziness_check}
    
    st.markdown("#### 증상 체크하기")
    col50, col51, col52, col53, col54, col55, col56, col57 = st.columns([0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]) # col 나누기
    with col50:
        st.error("증상 체크")
    with col51:
        cough_check = st.checkbox("기침")
        convulsion_check = st.checkbox("간헐적 경련")
    with col52:
        paralysis_check=st.checkbox('마비')
        nosense_check=st.checkbox('무감각')
    with col53:
        pain_check=st.checkbox('통증')
        touch_check=st.checkbox('만지면아픔')
    with col54:
        slowtalk_check=st.checkbox('말이 어눌해짐')
        swoon_check=st.checkbox('졸도')
    with col55:
        diarrhea_check=st.checkbox('설사')
        bleeding_check=st.checkbox('출혈')
    with col56:
        decreased_vision_check=st.checkbox('시력 저하')
        breathing_check=st.checkbox('호흡 곤란')
    with col57:
        dizziness_check=st.checkbox('현기증')

    
    ## -------------------------------------------------------------------------------------
    
    ## -------------------- ▼ 1-6그룹 중증 질환 여부, 중증 질환 판단(special_yn) col 구성 ▼ --------------------
    ## selectbox  사용(변수: special_yn)
    
    # col 나누기
    col60, col61 = st.columns([0.1, 0.3])
    with col60:
        st.error("중증질환여부")
    with col61:
        special_yn = st.selectbox('', ('중증질환아님', '중증질환예측', '중증질환선택'),
                                  label_visibility='collapsed')

    ##-------------------------------------------------------------------------------------
    
    ## -------------------- ▼ 1-7그룹 중증 질환 선택 또는 예측 결과 표시 cols 구성 ▼ --------------------
    
    col70, col71 = st.columns([0.1, 0.3])
    with col70:
        st.markdown("#### 예측 및 선택")
        
    with col71:
    
        if special_yn == "중증질환예측":

            patient_data = {
                "체온": [fever],
                "수축기 혈압": [high_blood],
                "이완기 혈압": [low_blood],
                "호흡 곤란": [breathing_check],
                "간헐성 경련": [convulsion_check],
                "설사": [diarrhea_check],
                "기침": [cough_check],
                "출혈": [bleeding_check],
                "통증": [pain_check],
                "만지면 아프다": [touch_check],
                "무감각": [nosense_check],
                "마비": [paralysis_check],
                "현기증": [dizziness_check],
                "졸도": [swoon_check],
                "말이 어눌해졌다": [slowtalk_check],
                "시력이 흐려짐": [decreased_vision_check],
                "중증질환": ""
            }
            
            # AI 모델 중증질환 예측 함수 호출
            st.session_state["special_m"] = predict_disease(patient_data)
            
            st.markdown(f"### 예측된 중증 질환은 {st.session_state.special_m}입니다")
            st.write("중증 질환 예측은 뇌출혈, 뇌경색, 심근경색, 응급내시경 4가지만 분류됩니다.")
            st.write("이외의 중증 질환으로 판단될 경우, 직접 선택하세요")

        elif special_yn == "중증질환선택":
            st.session_state["special_m"] = st.radio("중증 질환 선택", ['신생아', '중증화상', "사지접합",  "응급투석", "조산산모"], horizontal=True)
            st.markdown(f"### 예측된 중증 질환은 {st.session_state.special_m}입니다")

        else:
            st.session_state["special_m"] = "중증 아님"
            st.write(f"### 예측 결과는 {st.session_state.special_m}입니다")
            
    
    ## 완료시간 시간표시 cols 구성
    col180, col181 = st.columns([0.3, 0.7])
    with col180:
        st.info('완료 시간')
    with col181:
        my_end_time = datetime.time(hour = 0, minute = 0)
        end_time = st.time_input('', my_end_time, label_visibility='collapsed')
        
    ## -------------------- 완료시간 저장하기 START-------------------- 
    ## -------------------- ▼ 1-9그룹 완료시간 저장 폼 지정 ▼  --------------------    
    with st.form(key = 'tab2_second'):
    
        # 완료시간 저장 버튼
        if st.form_submit_button(label='저장하기'):
            dispatch_data = pd.read_csv('./119_emergency_dispatch.csv', encoding='cp949')
            id_num = list(dispatch_data['ID'].str[1:].astype(int))
            max_num = np.max(id_num)
            max_id = 'P' + str(max_num)
            elapsed = (end_time.hour - input_time.hour)*60 + (end_time.minute - input_time.minute)

            check_condition1 = (dispatch_data.loc[dispatch_data['ID'] == max_id, '출동일시'].values[0] == str(input_date))
            check_condition2 = (dispatch_data.loc[dispatch_data['ID'] == max_id, '이름'].values[0] == name)

            # 마지막 저장 내용과 동일한 경우, 내용을 update 시킴
            if check_condition1 and check_condition2:
                dispatch_data.loc[dispatch_data['ID'] == max_id, '나이'] = age
                dispatch_data.loc[dispatch_data['ID'] == max_id, '성별'] = patient_s
                dispatch_data.loc[dispatch_data['ID'] == max_id, '체온'] = fever
                dispatch_data.loc[dispatch_data['ID'] == max_id, '수축기 혈압'] = high_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이완기 혈압'] = low_blood
                dispatch_data.loc[dispatch_data['ID'] == max_id, '호흡 곤란'] = int(breath_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '간헐성 경련'] = int(convulsion_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '설사'] = int(diarrhea_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '기침'] = int(cough_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '출혈'] = int(bleeding_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '통증'] = int(pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '만지면 아프다'] = int(touch_pain_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '무감각'] = int(insensitive_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '마비'] = int(paralysis_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '현기증'] = int(dizziness_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '졸도'] = int(swoon_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '말이 어눌해졌다'] = int(inarticulate_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '시력이 흐려짐'] = int(blurred_check)
                dispatch_data.loc[dispatch_data['ID'] == max_id, '중증질환'] = st.session_state["special_m"]
                dispatch_data.loc[dispatch_data['ID'] == max_id, '이송 시간'] = int(elapsed)

            else: # 새로운 출동 이력 추가하기
                new_id = 'P' + str(max_num+1)
                new_data = {
                    'ID' : [new_id],
                    '출동일시' : [str(input_date)],
                    '이름' : [name],
                    '성별' : [patient_s],
                    '나이' : [age],
                    '체온': [fever],
                    '수축기 혈압': [high_blood],
                    '이완기 혈압': [low_blood],
                    '호흡 곤란': [int(breath_check)],
                    '간헐성 경련': [int(convulsion_check)],
                    '설사': [int(diarrhea_check)],
                    '기침': [int(cough_check)],
                    '출혈': [int(bleeding_check)],
                    '통증': [int(pain_check)],
                    '만지면 아프다': [int(touch_pain_check)],
                    '무감각': [int(insensitive_check)],
                    '마비': [int(paralysis_check)],
                    '현기증': [int(dizziness_check)],
                    '졸도': [int(swoon_check)],
                    '말이 어눌해졌다': [int(inarticulate_check)],
                    '시력이 흐려짐': [int(blurred_check)],
                    '중증질환': [st.session_state["special_m"]],
                    '이송 시간' : [int(elapsed)]
                }

                new_df= pd.DataFrame(new_data)
                dispatch_data = pd.concat([dispatch_data, new_df], axis=0, ignore_index=True)

            dispatch_data.to_csv('./119_emergency_dispatch.csv', encoding='cp949', index=False)
            
# -------------------- 완료시간 저장하기 END--------------------

## --------------------------------------------------------------------------


elif tabs == 'AED(심근경색)':
    st.markdown('## AED (자동심장충격기) 위치')
    with st.form(key='tab2_first'):
        # 병원 조회 버튼 생성
        if st.form_submit_button(label= 'AED 조회'):
            
            # 목적지 경위도
            destination_la, destination_lo = st.session_state.real_tmp.iloc[0, 5], st.session_state.real_tmp.iloc[0, 6]
            st.dataframe(st.session_state.real_tmp)
        
            
            # Antpath를 구현하기 위한 좌표 모음
            patient_point = str(st.session_state.patient_lo) + ',' + str(st.session_state.patient_la)
            destination_point = str(destination_lo) + ',' + str(destination_la)
            
            ## 확인용코드
            #st.write('출발지 좌표 : ', patient_point)
            #st.write('도착지 좌표 : ', destination_point)
            
            # 카카오모빌리티 API 실행
            road = kakao_road(patient_point=patient_point, destination_point=destination_point)
            #st.write(road)

            #### 지도에 표시
            with st.expander("인근 제세동기 리스트", expanded=True):
                
                
                # 지도 시각화
                m = folium.Map(location=[st.session_state.patient_la, st.session_state.patient_lo], zoom_start=16)
                icon = folium.Icon(color="red")
                folium.plugins.AntPath(locations = road, color = 'red').add_to(m)
                folium.Marker(location=[st.session_state.patient_la, st.session_state.patient_lo], 
                              popup="환자위치", tooltip="환자위치: "+st.session_state.location,
                              icon=icon).add_to(m)
                
                ###### folium을 활용하여 지도 그리기 (3일차 교재 branca 참조)
                for idx, row in st.session_state.real_tmp.iterrows():
                    html = """<!DOCTYPE html>
                    <html>
                        <table style="height: 126px; width: 330px;"> <tbody> <tr>
                        <td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">장소명</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['org'])+"""</tr>
                        <tr><td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">제세동기위치</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['buildPlace'])+"""</tr>
                        <tr><td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">거리(m)</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['거리'])+"""</tr>
                        <tr><td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">주소</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['buildAddress'])+"""</tr> 
                        <tr><td style="background-color: #2A799C;">
                    </tbody> </table> </html>"""
                    
                    iframe = branca.element.IFrame(html=html, width=350, height=175)
                    popup_text = folium.Popup(iframe,parse_html=True)
                    icon = folium.Icon(color="blue")
                    
                    folium.Marker(location=[row['wgs84Lat'], row['wgs84Lon']], 
                                  popup=popup_text, tooltip=row['buildAddress'], icon=icon).add_to(m)
                
                st_data = st_folium(m, width=1000)
        
        # # 문자보내기
        # if st.form_submit_button(label='문자 전송'):
        #     message = send_message(st.session_state.special_m, 
        #                            st.session_state.real_tmp.iloc[0, 2], 
        #                            st.session_state.real_tmp.iloc[0,-1])
        #     st.write(message)

# ============================================================================
   
elif tabs == '병원조회':
    st.markdown('## 병원조회')
    
    with st.form(key='tab3_first'):
        
        ### 병원 조회 버튼 생성
        if st.form_submit_button(label='병원조회'):

            #### 인근 병원 찾기 함수 호출
            hospital_list = find_hospital(st.session_state["special_m"], st.session_state.patient_la, st.session_state.patient_lo)
            
            #### 필요 병원 정보 추출 
            display_column = ['병원명', "주소", "응급연락처", "응급실수", "수술실수", "가용응급실수", "가용수술실수", '응급실포화도', '거리', '거리구분', '위도', '경도']
            tmp = hospital_list[display_column].sort_values(['거리구분', '응급실포화도', '거리'], ascending=[True, False, True])
            tmp.reset_index(drop=True, inplace=True)
        
            tmp_hospital = tmp.iloc[0:5, :]
            destination_lo, destination_la = tmp_hospital.iloc[0, -1], tmp_hospital.iloc[0, -2]
            # 확인용코드
            st.dataframe(tmp_hospital)
            
            # Antpath를 구현하기 위한 좌표 만들기
            patient_point = (st.session_state.patient_lo, st.session_state.patient_la)
            destination_point = (destination_lo, destination_la)
            # 확인용코드
            #st.write('출발지 좌표 : ', patient_point)
            #st.write('도착지 좌표 : ', destination_point)
            
            # kakao 최단경로 api 호출
            road = kakao_road(patient_point=patient_point, destination_point=destination_point)
            

            #### 추출 병원 지도에 표시
            with st.expander("인근 병원조회", expanded=True):

                # folium 지도 출력
                m = folium.Map(location=[st.session_state.patient_la, st.session_state.patient_lo], zoom_start=13)
                icon = folium.Icon(color="red")
                folium.Marker(location=[st.session_state.patient_la, st.session_state.patient_lo], popup="환자위치", 
                              tooltip="환자위치 : " + st.session_state['location'], icon=icon).add_to(m)

                
                ###### folium을 활용하여 지도 그리기 (branca)
                
                for idx, row in hospital_list.iterrows():
                    html = """<!DOCTYPE html>
                    <html>
                        <table style="height: 126px; width: 330px;"> <tbody> <tr>
                        <td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">병원명</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['병원명'])+"""</tr> 
                        <tr><td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">거리(km)</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['거리'])+"""</tr>
                        <tr><td style="background-color: #2A799C;">
                        <div style="color: #ffffff;text-align:center;">응급실포화도</div></td>
                        <td style="width: 230px;background-color: #C5DCE7;">{}</td>""".format(row['응급실포화도'])+"""</tr>
                    </tbody> </table> </html>"""
                    
                    iframe = branca.element.IFrame(html=html, width=350, height=150)
                    popup_text = folium.Popup(iframe,parse_html=True)
                    icon = folium.Icon(color="blue")
                    
                    folium.Marker(location=[row['위도'], row['경도']], 
                                  popup=popup_text, tooltip=row['병원명'], icon=icon).add_to(m)
                    
                # AntPath 만들기
                folium.plugins.AntPath(locations = road, color = 'red').add_to(m)
                st_data = st_folium(m, width=1000)
                
        # # 문자보내기
        # if st.form_submit_button(label='문자 전송'):
        #     message = send_message(st.session_state.special_m, 
        #                            hospital_tmp.iloc[0, 0], 
        #                            hospital_tmp.iloc[0, -4])
        #     st.write(message)

    ## ------------------------------------------------------------------------------


elif tabs == '통계':

    # 제목 넣기
    st.markdown('### 광주광역시 응급상황 통계')

    # folium map 생성
    m = folium.Map(location=['35.15574', '126.83543 '], # 광주광역시 중심좌표
                   zoom_start=11,
                   tiles='OpenStreetMap')

    # Choropleth 맵 생성
    folium.Choropleth(

        geo_data = gwangju_grid,                      # Choropleth 맵에 사용될 지리 정보
        data = gwangju_grid,                          # Choropleth 맵에 사용될 데이터
        columns = ['gid', 'emergency'],               # 데이터에서 사용될 열 이름 
        key_on='feature.properties.gid',              # GeoJson에서 고유 식별자로 사용할 (지리적 영역을 구분할) 열 이름   

        fill_color='Reds',                  # 색상 팔레트 설정  예) 'YlGnBu', 'YlOrRd'
        fill_opacity= 0.7,                  # 채우기 투명도 설정 / 0 (투명) ~ 1 
        bins= 100,                          # 색 구간

        line_color = 'black',               # 경계선 색
        line_opacity = 0.5,                 # 경계선 투명도

        nan_fill_color = 'white',           # 결측치 색
        nan_fill_opacity = 0,               # 결측치 투명도 - 0 (완전투명)

        name='광주 응급현황',                # 이름 
        legend_name='광주 응급현황',         # 범례 이름 설정
        highlight=True,                     # 마우스 갖다대면 강조됨
    ).add_to(m)

    # tooltip 설정
    folium.GeoJson(
        gwangju_grid,
        tooltip=folium.GeoJsonTooltip(fields=['읍면동', 'emergency'], aliases=['법정동 :', '응급상황 :']),
        style_function=lambda feature: {'fillColor': 'transparent', 'color': 'transparent'}
    ).add_to(m)

    st_data = st_folium(m, width = 1100, height = 600)

    
    ## -------------------------------------------------------------------

    ## -------------------- ▼ 2-1그룹 통계 조회 기간 선택하기 ▼ --------------------
    
    col210, col211, col212 = st.columns([0.3, 0.2, 0.1])
    
    with col210:
        slider_date = st.slider('날짜', min_value = min_date, max_value = max_date,
                               value=(min_date, now_date2))
    with col211:
        slider_week = st.slider('주간', min_value = min_date, max_value = max_date,
                               value=(min_date, now_date2), step = datetime.timedelta(weeks=1))
    with col212:
        slider_month = st.slider('월간', min_value = min_date, max_value = max_date,
                               value=(min_date, now_date2), step = datetime.timedelta(weeks=1),
                                format = 'YYYY-MM')
    
    ## 선택된 일자의 data 추출
    data['datetime'] = pd.to_datetime(data['출동일시'])
    day_list_df = data[(slider_date[0] <= data['datetime']) & (slider_date[1] >= data['datetime'])]

    ## 선택된 주간의 data 추출
    data['week'] = data['datetime'].dt.strftime("%W").astype(int)
    min_week = int(slider_week[0].strftime("%W"))
    max_week = int(slider_week[1].strftime("%W"))
    week_list_df = data[(data['week'] >= min_week) & (data['week'] <= max_week)]
        
    ## 선택된 월의 data 추출
    data['month'] = data['datetime'].dt.month.astype(int)
    min_month = slider_month[0].month
    max_month = slider_month[1].month
    month_list_df = data[(data['month'] >= min_week) & (data['month'] <= max_week)]

    ## ------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-2그룹 일간/주간/월간 총 출동 건수 통계 그래프 ▼ --------------------
  
    select_bins = st.radio('주기', ('일자별', '주별', '월별'), horizontal=True)
    if select_bins == '일자별':
        
        group_day = day_list_df.groupby(by='datetime', as_index=False)['ID'].count()
        group_day = group_day.rename(columns={'ID' : '출동건수'})
        st.bar_chart(data=group_day, x='datetime', y='출동건수', use_container_width=True)

    elif select_bins == '주별':

        group_week = week_list_df.groupby(by='week', as_index=False)['ID'].count()
        group_week = group_week.rename(columns={'ID' : '출동건수'})
        group_week = group_week.sort_values('week', ascending=True)
        st.bar_chart(data=group_week, x='week', y='출동건수', use_container_width=True)

    else:

        group_month = month_list_df.groupby(by='month', as_index=False)['ID'].count()
        group_month = group_month.rename(columns={'ID' : '출동건수'})
        group_month = group_month.sort_values('month', ascending=True)
        st.bar_chart(data=group_month, x='month', y='출동건수', use_container_width=True)

    ## -------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-3그룹 일간/주간/월간 평균 이송시간 통계 그래프 ▼ --------------------
    
    st.info('이송시간 통계')
    col230, col231, col232 = st.columns(3)

    with col230:

        group_day_time = day_list_df.groupby(by=['출동일시'], as_index=False)['이송 시간'].mean()
        group_day_time = group_day_time.rename(columns={'이송 시간': '이송 시간'})
        st.line_chart(data=group_day_time, x='출동일시', y='이송 시간', use_container_width=True)

    with col231:

        group_week_time = week_list_df.groupby(by=['나이'], as_index=False)['이송 시간'].mean()
        group_week_time = group_week_time.rename(columns={'이송 시간': '이송 시간'})
        st.line_chart(data=group_week_time, x='나이', y='이송 시간', use_container_width=True)

    with col232:

        group_month_time = month_list_df.groupby(by=['중증질환'], as_index=False)['이송 시간'].mean()
        group_month_time = group_month_time.rename(columns={'이송 시간': '이송 시간'})
        st.line_chart(data=group_month_time, x='중증질환', y='이송 시간', use_container_width=True)
    
    ## -------------------------------------------------------------------------------------------

    ## -------------------- ▼ 2-4그룹 일간/주간/월간 중증 질환별 비율 그래프 ▼ --------------------
    
    st.info('중증 질환별 통계')

    col240, col241, col242 = st.columns(3)
    
    with col240:
    
        group_day_disease = day_list_df.groupby(by='중증질환', as_index=False)['datetime'].count()
        group_day_disease = group_day_disease.rename(columns={'datetime':'count'})

        fig = px.pie(group_day_disease, values='count', names='중증질환', height=360, width=360, hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title='일간 통계')
        st.plotly_chart(fig)

    with col241:

        group_week_disease = week_list_df.groupby(by='중증질환', as_index=False)['week'].count()
        group_week_disease = group_week_disease.rename(columns={'week':'count'})

        fig = px.pie(group_week_disease, values='count', names='중증질환', height=360, width=360, hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title='주간 통계')
        st.plotly_chart(fig)

    with col242:

        group_month_disease = month_list_df.groupby(by='중증질환', as_index=False)['month'].count()
        group_month_disease = group_month_disease.rename(columns={'month':'count'})

        fig = px.pie(group_month_disease, values='count', names='중증질환', height=360, width=360, hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title='월간 통계')
        st.plotly_chart(fig)
