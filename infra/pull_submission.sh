#!/usr/bin/env bash
# pod의 제출 CSV를 로컬 submissions/로 수거 (base64, scp 불가 우회).
# 사용: bash infra/pull_submission.sh submission_v4.csv
# 원리: pod에서 base64 인코딩 → 본문 출력 → 로컬에서 "순수 base64 줄만" 필터 후 디코드 → md5/행수 대조.
#   (pod.sh가 명령을 PTY로 3번 에코하므로, 마커 대신 base64 알파벳 only 줄만 추출해 오염 차단)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="${1:?수거할 CSV 파일명 필요 (예: submission_v4.csv)}"
REMOTE="/workspace/dacon-bias-challenge/submissions/$NAME"
OUTDIR="$HERE/../submissions"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/$NAME"

# 1) pod에서 base64(76칼럼 wrap) 인코딩 + md5/행수 기록 (단일 거대라인은 PTY에서 깨지므로 wrap 사용)
echo "base64 '$REMOTE' > /workspace/_pull.b64 2>/dev/null; md5sum '$REMOTE' 2>/dev/null | cut -d' ' -f1 > /workspace/_pull.md5; wc -l < '$REMOTE' > /workspace/_pull.lines; echo PREP_DONE" \
  | POD_TIMEOUT=120 bash "$HERE/pod.sh" >/dev/null 2>&1

# 2) b64 본문 받기 — 공백/특수문자 없는 '순수 base64 줄'만 추출(명령 에코·배너·마커 자동 제거)
# 패딩(앞줄 잘림 흡수) + 분할마커로 본문 구간만 awk 추출 후 base64-charset 필터 → 디코드.
#   마커는 명령 에코엔 M64''STA 로 보여 매칭 안 되고, 실제 실행 출력 M64STA 만 awk가 잡음.
echo "for i in \$(seq 1 10); do echo P_\$i; done; echo M64''STA; cat /workspace/_pull.b64; echo M64''END" \
  | POD_TIMEOUT=180 bash "$HERE/pod.sh" 2>/dev/null | tr -d '\r' \
  | awk '/M64STA/{f=1;next} /M64END/{f=0} f' \
  | grep -aE '^[A-Za-z0-9+/=]+$' | tr -d '\n' | base64 -d > "$OUT" 2>/dev/null

# 3) md5/행수 대조 (원격 값은 숫자/16진만 매칭되어 명령 에코와 충돌 없음)
INFO=$(echo "for i in \$(seq 1 6); do echo pad\$i; done; echo RMD5=\$(cat /workspace/_pull.md5); echo RLINES=\$(cat /workspace/_pull.lines)" \
  | POD_TIMEOUT=60 bash "$HERE/pod.sh" 2>/dev/null | tr -d '\r')
REMOTE_MD5=$(echo "$INFO" | grep -aoE 'RMD5=[0-9a-f]{32}' | tail -1 | cut -d= -f2)
REMOTE_LINES=$(echo "$INFO" | grep -aoE 'RLINES=[0-9]+' | tail -1 | cut -d= -f2)
LOCAL_MD5=$(md5sum "$OUT" | cut -d' ' -f1)
LOCAL_LINES=$(wc -l < "$OUT")

echo "수거: $OUT ($(wc -c < "$OUT") bytes)"
echo "  로컬 md5=$LOCAL_MD5 (행 $LOCAL_LINES) / 원격 md5=$REMOTE_MD5 (행 $REMOTE_LINES)"
if [ -n "$REMOTE_MD5" ] && [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
  echo "  OK 무결성 일치 — DACON 업로드 준비 완료"
else
  echo "  WARN md5 불일치/미확인 — 재수거 필요"
fi
