from dataclasses import dataclass

@dataclass
class AttackConfig:
    attempts: int = 2
    seed: int = 0
    use_mutation: bool = True
