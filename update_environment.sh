#!/bin/bash


echo "==================================================="
echo "  Обновление окружения NeuroDetekt"
echo "==================================================="

if ! command -v conda &> /dev/null; then
    echo "❌ Conda не найдена. Установите Anaconda или Miniconda."
    exit 1
fi

if conda env list | grep -q "neurodetekt"; then
    echo "✅ Окружение neurodetekt найдено."
    
    conda_path=$(conda info --base)
    source "$conda_path/etc/profile.d/conda.sh"
    conda activate neurodetekt
    
    echo "🔄 Обновление окружения..."
    conda env update -f environment.yml
    
    echo "🔄 Установка совместимой версии NumPy..."
    conda install -y numpy=1.24.3
    
    echo "✅ Окружение успешно обновлено."
else
    echo "⚠️ Окружение neurodetekt не найдено. Создаем новое окружение..."
    conda env create -f environment.yml
    
    echo "✅ Окружение успешно создано."
fi

echo ""
echo "Для активации окружения выполните:"
echo "conda activate neurodetekt"
echo ""
echo "Затем проверьте окружение командой:"
echo "python src/check_environment.py"
echo "" 