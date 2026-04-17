#!/bin/bash

# Script per creare .netrc file dalle variabili d'ambiente
# Per autenticazione al registry Azure DevOps

# Controlla se le variabili d'ambiente necessarie sono presenti
if [ -z "$MYWAI_ARTIFACTS_MAIL" ]; then
    echo "Errore: MYWAI_ARTIFACTS_MAIL non è definita"
    exit 1
fi

if [ -z "$MYWAI_ARTIFACTS_TOKEN" ]; then
    echo "Errore: MYWAI_ARTIFACTS_TOKEN non è definita"
    exit 1
fi

# Determina la directory root del progetto (dove si trova pyproject.toml)
# Se lo script è in scripts/linux/, la root è due livelli sopra
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Crea il contenuto del file .netrc nella home directory (dove uv lo cerca)
# Route del progetto: pkgs.dev.azure.com/zenatek-mywai/_packaging/zenatek-mywai/pypi/simple/
# uv cerca il file .netrc nella home directory (~/.netrc) di default
NETRC_FILE="$HOME/.netrc"

# Crea anche una copia nella root del progetto per riferimento
PROJECT_NETRC_FILE="$PROJECT_ROOT/.netrc"

echo "HOME=$HOME"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "NETRC_FILE=$NETRC_FILE"

# Crea il file .netrc nella home directory (dove uv lo cerca)
cat > "$NETRC_FILE" << EOF
machine pkgs.dev.azure.com
    login $MYWAI_ARTIFACTS_MAIL
    password $MYWAI_ARTIFACTS_TOKEN
EOF

# Imposta i permessi corretti (600 = solo proprietario può leggere/scrivere)
chmod 600 "$NETRC_FILE"

# Crea anche una copia nella root del progetto per riferimento
cp "$NETRC_FILE" "$PROJECT_NETRC_FILE"
chmod 600 "$PROJECT_NETRC_FILE"

echo ".netrc file creato con successo!"
echo "File creato in home directory: $NETRC_FILE (uv lo userà automaticamente)"
echo "Copia creata anche in: $PROJECT_NETRC_FILE"
echo "File creato per autenticazione al registry Azure DevOps"

