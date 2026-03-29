"""Terminal UI helpers for NovelWritingAgent."""

from __future__ import annotations

from dataclasses import dataclass


class Colors:
    """ANSI color definitions."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


@dataclass(slots=True)
class NovelUI:
    """Small terminal renderer for progress and previews."""

    width: int = 74

    def banner(self, title: str, subtitle: str | None = None) -> str:
        lines = [
            f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╔{'═' * self.width}╗{Colors.RESET}",
            self._line(title, color=Colors.BRIGHT_WHITE, bold=True),
        ]
        if subtitle:
            lines.append(self._line(subtitle, color=Colors.DIM))
        lines.append(f"{Colors.BOLD}{Colors.BRIGHT_CYAN}╚{'═' * self.width}╝{Colors.RESET}")
        return "\n".join(lines)

    def section(self, label: str) -> str:
        top = f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}┌{'─' * self.width}┐{Colors.RESET}"
        mid = self._boxed_line(f" {label}", border_color=Colors.BRIGHT_YELLOW, text_color=Colors.BRIGHT_YELLOW, bold=True)
        bot = f"{Colors.BOLD}{Colors.BRIGHT_YELLOW}└{'─' * self.width}┘{Colors.RESET}"
        return "\n".join([top, mid, bot])

    def event(self, label: str, message: str, color: str = Colors.BRIGHT_CYAN) -> str:
        chip = f"{Colors.BOLD}{color}[{label.upper():^10}]{Colors.RESET}"
        return f"{chip} {Colors.WHITE}{message}{Colors.RESET}"

    def preview(self, title: str, text: str, color: str = Colors.BRIGHT_BLUE, max_lines: int = 8) -> str:
        body_lines = [line.rstrip() for line in text.strip().splitlines() if line.strip()]
        body = body_lines[:max_lines]
        if len(body_lines) > max_lines:
            body.append("...")
        rendered = [
            f"{Colors.BOLD}{color}┌{'─' * self.width}┐{Colors.RESET}",
            self._boxed_line(f" {title}", border_color=color, text_color=color, bold=True),
            f"{Colors.BOLD}{color}├{'─' * self.width}┤{Colors.RESET}",
        ]
        for line in body:
            rendered.append(self._boxed_line(f" {line}", border_color=color, text_color=Colors.DIM))
        rendered.append(f"{Colors.BOLD}{color}└{'─' * self.width}┘{Colors.RESET}")
        return "\n".join(rendered)

    def summary(self, title: str, items: list[str], color: str = Colors.BRIGHT_GREEN) -> str:
        rendered = [
            f"{Colors.BOLD}{color}┌{'─' * self.width}┐{Colors.RESET}",
            self._boxed_line(f" {title}", border_color=color, text_color=color, bold=True),
            f"{Colors.BOLD}{color}├{'─' * self.width}┤{Colors.RESET}",
        ]
        for item in items:
            rendered.append(self._boxed_line(f" {item}", border_color=color, text_color=Colors.WHITE))
        rendered.append(f"{Colors.BOLD}{color}└{'─' * self.width}┘{Colors.RESET}")
        return "\n".join(rendered)

    def _line(self, text: str, color: str = Colors.WHITE, bold: bool = False) -> str:
        styled = f"{color}{text}{Colors.RESET}"
        if bold:
            styled = f"{Colors.BOLD}{styled}"
        inner_width = self.width
        plain = text[:inner_width]
        padding = max(0, inner_width - len(plain))
        return f"{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}{styled}{' ' * padding}{Colors.BOLD}{Colors.BRIGHT_CYAN}║{Colors.RESET}"

    def _boxed_line(
        self,
        text: str,
        border_color: str,
        text_color: str,
        bold: bool = False,
    ) -> str:
        plain = text[: self.width]
        padding = max(0, self.width - len(plain))
        styled = f"{text_color}{plain}{Colors.RESET}"
        if bold:
            styled = f"{Colors.BOLD}{styled}"
        return f"{Colors.BOLD}{border_color}│{Colors.RESET}{styled}{' ' * padding}{Colors.BOLD}{border_color}│{Colors.RESET}"
