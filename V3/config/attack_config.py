from dataclasses import dataclass


@dataclass
class AttackConfig:
    use_misleading_comments: bool = False
