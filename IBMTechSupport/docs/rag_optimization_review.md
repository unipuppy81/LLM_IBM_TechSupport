# RAG 성능 최적화 5단계 실험 결과

## 1. 기존 Baseline 성능

평가 데이터: 총 135개
- 답변 가능: 90개
- 답변 불가능: 45개

평가 결과:
- Token F1: 16.52%
- Retrieval Hit@5: 66.67% (60/90)
- 잘못된 답변 거부: 45/90개
- 정상 답변 거부: 29/45개

## 2. Evidence Judge 사용 여부 비교

Judge 사용:
- Token F1: 16.52%
- 정상 답변 거부: 29개
- 잘못된 답변 거부: 45개

Judge 미사용:
- Token F1: 28.79%
- 정상 답변 거부: 1개
- 잘못된 답변 거부: 0개

결론:
Judge 제거 시 F1은 상승하지만 답변 불가능한 질문에도
대부분 답변하므로 기존 Judge 유지.

## 3. Judge 프롬프트 및 모델 비교

Baseline 1.5B:
- 잘못된 거부: 45개
- 정상 거부: 29개

Balanced 1.5B:
- 잘못된 거부: 67개
- 정상 거부: 37개

Strict 1.5B:
- 잘못된 거부: 55개
- 정상 거부: 35개

Strict 3B:
- Token F1: 14.11%
- 잘못된 거부: 50개
- 정상 거부: 34개

결론:
프롬프트 변경과 모델 크기 증가 모두
기존 Baseline보다 전반적인 성능 개선에 실패.
기존 1.5B Judge 유지.

## 4. 주변 청크 확장 실험

Radius 0:
- Token F1: 16.52%

Radius 1:
- Token F1: 15.40%

Radius 2:
- Token F1: 14.52%

결론:
문맥을 확장할수록 처리 시간이 증가하고
Token F1은 하락.
기존 Radius 0 유지.

## 5. 제목 기반 Contextual Retrieval

기존 검색:
- Hit@5: 66.67%

Title BM25:
- Hit@5: 66.67%

Title BM25 + Dense:
- Hit@5: 62.22%

결론:
문서 제목 추가로 검색 성능 개선 실패.
기존 검색 방식 유지.

## 6. Generation 프롬프트 최적화

Baseline:
- Token F1: 16.52%

Concise:
- Token F1: 16.16%

Evidence Check:
- Token F1: 16.16%

결론:
프롬프트 변경으로 답변 품질 개선 실패.
기존 Generation 프롬프트 유지.

## 7. 전체 결론

5단계 최적화 실험에서 기존 Baseline보다
우수한 설정을 확인하지 못함.

기존 RAG 파이프라인 유지.

현재 확인된 주요 문제:
1. Retrieval에서 정답 출처 검색 실패 30개
2. 정답 출처 검색 후 Judge 전체 거부 26개
3. 답변 불가능 질문에 답변 생성 16개
4. 답변 생성 대상 평균 Token F1 33.04%

## 8. 다음 개발 작업

1. Retrieval 실패 30개 원인 분석
2. 실제 정답 근거가 포함된 청크 검증
3. Reranker 후보 수 및 청크 제한 분석
4. Evidence Judge 오판 사례 분석
5. Judge P(YES)와 Reranker 점수 비교
6. Reranker 도메인 Fine-tuning
7. 근거 문장 추출형 Generation 실험
8. 최종 성능 비교 및 검증
