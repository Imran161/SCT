# Используем официальный образ Alpine
FROM alpine:edge

# Устанавливаем необходимые зависимости
RUN apk add --no-cache \
    neovim \
    git \
    curl \
    wget \
    build-base \
    ripgrep \
    lazygit \
    python3 \
    py3-pip

# Создаем рабочую директорию
WORKDIR /home/lazyvimuser

# Устанавливаем LazyVim
RUN git clone https://github.com/LazyVim/starter ~/.config/nvim

# Устанавливаем рабочую директорию как volume
VOLUME ["/home/lazyvimuser"]

# Запускаем Neovim
CMD ["nvim"]
