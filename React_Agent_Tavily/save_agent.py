from datetime import datetime

filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
with open(filename, "w", encoding="utf-8") as f:
    f.write(f"# Исследовательский отчёт: {topic}\n\n")
    f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write(f"**Инструмент:** Tavily Research + LlamaIndex\n\n")
    f.write("---\n\n")
    f.write(response.response)