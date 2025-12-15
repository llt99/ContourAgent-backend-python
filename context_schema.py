from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class MCPContextSchema:
    task: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    feedbackParsed: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    run_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def update_params(self, new_params: Dict[str, Any]):
        """更新 params 部分"""
        self.params.update(new_params)

    def add_error(self, error_msg: str):
        """记录错误"""
        self.errors.append(error_msg)

    def add_history(self, record: Dict[str, Any]):
        """记录操作历史"""
        self.history.append(record)
