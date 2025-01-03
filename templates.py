rag_template = """
You are an expert in government policies regarding the startup ecosystem in South Korea.

Given the relevant context, answer the user's query appropriately, stick with the data from the context.
Make sure to generate your response in korean.

Also, below is the information which can be helpful for you to understand the context better:

<Information regarding context>
[중소벤처기업부 공고 제2024-626호]
2025년 중앙부처 및 지자체 창업지원사업 통합공고
｢중소기업창업 지원법｣ 제14조(창업정책정보의 수집 및 제공)에 따라 2025년 창업지원사업을 공고하오니, 창업기업 및 예비창업자의 적극 적인 참여를 바랍니다.
2024년 12월 31일
중소벤처기업부장관

◦	근거 :  중소기업창업 지원법 	제14조
◦	목적 : 창업자 및 예비창업자가 국내 창업지원사업 정보를 알기 쉽게 접할 수 있도록 중앙부처 및 지자체 창업지원사업 통합공고
◦	경과
-	(‘16) 중앙부처 창업지원사업 통합공고 실시(6개 기관, 65개 사업, 0.6조원)
-	(‘21) 광역지자체 창업지원사업 추가(31개 기관, 193개 사업, 1.5조원)
-	(‘22) 기초지자체 창업지원사업 및 융자 사업 추가
-	(‘23) 창업지원사업 8개 유형별로 구분
* 사업화, 시설·공간·보육, 멘토링·컨설팅·교육, 행사·네트워크, 글로벌 진출, 융자, 기술 개발(R&D), 인력
< ’21~‘25년 창업지원사업 통합공고 현황>
구분	참여기관	사업규모
‘21년	14개 중앙부처
17개 광역지자체	89개 사업, 13,812억원(중앙부처)
104개 사업, 811억원(광역지자체)
‘22년	14개 중앙부처
17개 광역지자체
63개 기초지자체	100개 사업, 35,578억원(중앙부처)
152개 사업, 885억원(광역지자체)
126개 사업, 205억원(기초지자체)
‘23년	14개 중앙부처
17개 광역지자체
72개 기초지자체	102개 사업, 35,078억원(중앙부처)
176개 사업, 1,243억원(광역지자체)
148개 사업, 286억원(기초지자체)
‘24년	11개 중앙부처
17개 광역지자체
71개 기초지자체	86개 사업, 35,621억원(중앙부처)
160개 사업, 1,167억원(광역지자체)
151개 사업, 283억원(기초지자체)
‘25년	13개 중앙부처
17개 광역지자체
88개 기초지자체	87개사업, 31,190억원(중앙부처)
170개 사업, 1,338억원(광역지자체)
172개 사업, 412억원(기초지자체)
 
   
□ 사업 현황
(1)	총괄
◦	대상기관․사업 : 101개 기관, 429개 사업
◦	예산규모 : 3조 2,940억원
< 연도별 통합공고 현황(단위 : 개, 억원) >
구분	’21년	’22년	‘23년	‘24년(A)	‘25년(B)
지원기관(개)	31	94	103	99	101
	중앙부처	14	14	14	11	13
	광역지자체	17	17	17	17	17
	기초지자체	-	63	72	71	71
대상사업(개)	193	378	426	397	429
	중앙부처	89	100	102	86	87
	광역지자체	104	152	176	160	170
	기초지자체	-	126	148	151	172
지원예산(억원)	14,623	36,668	36,607	37,121	32,940
	중앙부처	13,812	35,578	35,078	35,621	31,190
	광역지자체	811	885	1,243	1,167	1,338
	기초지자체	-	205	286	333	412
(2)	사업유형별 현황
◦	사업유형별 : 융자(1조 5,552억원, 12개), 사업화(7,666억원, 169개), 기 술개발(6,292억원, 8개), 시설·공간·보육(1,501억원, 123개), 글로벌 진출(1,233억원, 21개) 등으로 구성
<분야별 창업지원사업 수 및 예산 현황(단위 : 억원, %, 개)>
구분	융자	사업화	기술개발(R&D)	시설·공간·보육	글로벌 진출
예산	15,552	7,666	6,292	1,502	1,233
(비율)	47.1	23.3	19.1	4.6	3.7
사업수	12	169	8	123	21
(비율)	2.8	39.4	1.9	28.7	4.9
구분	멘토링·컨설팅‧교육	행사·네트워크	인력	합계	 
예산	394	268	33	32,940	
(비율)	1.2	0.8	0.1	100.0	
사업수	56	37	3	429	
(비율)	13.1	8.6	0.7	100.0	
 
(3)	기관별 현황
◦	(중앙부처) 13개 기관, 87개 사업, 3조 1,190억원
- 중기부(2조 9,499억원, 94.6%), 문체부(530억원), 과기부(454억원) 순
◦	(지자체) 88개 기관, 342개 사업, 1,750억원
- 서울시(382억원, 21.8%), 경기도(200억원), 경남도(186억원) 순
< 창업지원사업 현황(단위: 개, 억원, %) >
구 분	중앙부처	지자체(광역+기초)
	기관명	사업수	예산(비율)	기관명	기관수	사업수	예산(비율)
1	중기부	35	29,499	94.6	서울	14	36	382	21.8
2	문체부	11	530	1.7	경기	14	54	200	11.4
3	과기부	13	454	1.5	경남	10	44	186	10.6
4	환경부	3	237	0.8	전북	6	29	139	7.9
5	농식품부	7	230	0.7	충남	2	9	136	7.8
6	특허청	2	113	0.4	대전	3	15	106	6.1
7	교육부	1	21	0.1	광주	5	16	105	6.0
8	해수부	2	34	0.1	부산	4	22	104	5.9
9	복지부	4	24	0.1	제주	1	20	103	5.9
10	법무부	1	14	0.1	강원	11	25	61	3.5
11	국토부	2	13	0.1	충북	1	18	58	3.3
12	통일부	3	8	0.1	경북	4	11	44	2.5
13	방사청	3	13	0.1	울산	3	8	42	2.4
14					인천	4	15	30	1.7
15					전남	4	8	25	1.4
16					대구	1	6	22	1.3
17					세종	1	6	7	0.4
계	13	87	31,190	100.0		88	342	1,750	100.0

□	신청방법
◦	K-Startup 포털(k-startup.go.kr) 및 각 기관 홈페이지 등을 통해 사업별 별도 공고예정으로, 신청자격 등을 확인하여 개별 신청
</Information regarding context>

context: {context}

user query: {query}
"""

competitor_template = """
You are an expert in understanding different business in Seoul.
Given a user's query identify the major competitors of the business, company or the industry mentioned in the user's query.
Mention the competitors and details about them in your response
Generate your response in korean.
context: {context}

user query: {query}
"""