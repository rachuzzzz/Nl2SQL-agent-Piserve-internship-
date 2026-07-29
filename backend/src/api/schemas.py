"""
Pydantic request/response models for the FastAPI API.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class PageRequest(BaseModel):
    session_id: str
    page:       int           # 0-based page index
    page_size:  int = Field(default=50, ge=1, le=500)

class ChartRequest(BaseModel):
    session_id: str
    chart_type: str | None = None   # None → auto-advise; string → force that type
    x_col:      str | None = None   # override label column (column picker)
    y_cols:     list[str] | None = None  # override value columns (column picker)
    view_all:   bool = False   # drill-down — skip top-N aggregation, return the full fetched dataset

class StepInfo(BaseModel):
    tool: str
    thought: str
    sql: str | None = None
    result_preview: str | None = None
    success: bool = True

class DisambiguationMeta(BaseModel):
    kind: str          # "person" | "facility" | "project" | "client" | "organisation"
    search_term: str   # literal text from the STEP 1 ILIKE clause, e.g. "adnoc"

class ResultData(BaseModel):
    rows: list[dict]
    columns: list[str]
    row_count: int
    total_count: int | None = None
    page: int = 0
    page_size: int = 50
    has_more: bool = False
    sql: str | None = None   # base SQL — sent back with every /api/page call
    disambiguation: DisambiguationMeta | None = None

class ChartConfig(BaseModel):
    chart_type: str | None
    reason: str
    confidence: str
    config: dict | None = None
    conflict: bool = False
    requested_type: str | None = None
    requested_config: dict | None = None
    alternatives: list[dict] = []
    x_label: str | None = None      # category / x-axis column name
    y_labels: list[str] = []        # value column name(s)
    # Top-N truncation metadata — mirrors ResultData's total_count/truncated
    # pattern. For aggregated chart types (bar/pie/radar/... — see
    # charts/advisor/public.py's _TOPN_BY_TYPE) rows_shown/rows_total count
    # CATEGORIES (post-aggregation), not raw rows, since that's what's
    # visually truncated. rows_shown == rows_total when nothing was truncated.
    rows_shown: int | None = None
    rows_total: int | None = None
    is_aggregated: bool = False
    can_view_all: bool = False      # true when a fuller/un-aggregated fetch is available via ChartRequest.view_all

class DashboardData(BaseModel):
    show: bool = False
    cards: list[dict] = []   # [{label, value, kind, icon}]

class ChatResponse(BaseModel):
    answer: str
    sql: str | None
    reasoning: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_ms: int
    llm_calls: int
    sql_errors: int
    steps: list[StepInfo]
    session_id: str
    result_data: ResultData | None = None
    chart_config: ChartConfig | None = None
    dashboard: DashboardData | None = None


class TTSRequest(BaseModel):
    text: str
