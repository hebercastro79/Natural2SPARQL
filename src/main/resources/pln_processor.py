import nltk
import spacy
import difflib
import re
import unicodedata
from nltk.stem.rslp import RSLPStemmer
import logging
import sys
import json
import os
from datetime import datetime, timedelta
import io # Para forçar encoding

# --- Configuração de Logging (Arquivo e Stderr) ---
# Remover manipuladores padrão para evitar duplicidade ou saída no stdout
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Formato do Log
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Manipulador para arquivo 'pln_processor.log'
try:
    file_handler = logging.FileHandler('pln_processor.log', encoding='utf-8', mode='w') # 'w' para sobrescrever a cada execução
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # Captura DEBUG e acima no arquivo
    logging.root.addHandler(file_handler)
except Exception as e:
    # Se não conseguir criar o log, imprime no stderr original e continua
    print(f"AVISO URGENTE: Não foi possível criar/abrir o arquivo de log 'pln_processor.log'. Erro: {e}", file=sys.__stderr__)

# Manipulador para stderr (console do Python, NÃO o stdout lido pelo Java)
# Mostra apenas WARNING e acima no console para não poluir muito
try:
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(log_formatter)
    stderr_handler.setLevel(logging.WARNING) # Apenas WARNING, ERROR, CRITICAL no console
    logging.root.addHandler(stderr_handler)
except Exception as e:
    print(f"AVISO: Não foi possível configurar logging para stderr. Erro: {e}", file=sys.__stderr__)


# Define o nível raiz para o mais baixo dos handlers (DEBUG)
logging.root.setLevel(logging.DEBUG)

# Tentar forçar UTF-8 para stdout/stderr com errors='replace'
try:
    # Usar buffer para evitar problemas com terminais que não suportam reconfiguração direta
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    logging.info("Encoding UTF-8 (com replace) forçado para stdout/stderr.")
except Exception as e:
    logging.warning(f"Não foi possível forçar UTF-8 em stdout/stderr (pode ser normal em alguns ambientes): {e}")
# -----------------------------------------------------

# --- Carregamento de Modelos e Dados ---
try:
    nltk.download('punkt', quiet=True, raise_on_error=True)
    nltk.download('rslp', quiet=True, raise_on_error=True)
    logging.info("Pacotes NLTK verificados/baixados.")
except Exception as e:
    logging.error(f"Erro ao baixar pacotes NLTK: {e}", exc_info=True)
    print(json.dumps({"erro": "Erro ao inicializar NLTK.", "detalhes": str(e)}, ensure_ascii=False)) # Erro JSON para Java (stdout)
    print(f"Erro fatal ao baixar pacotes NLTK: {e}", file=sys.stderr)
    sys.exit(1)

try:
    nlp = spacy.load("pt_core_news_sm")
    logging.info("Modelo spaCy 'pt_core_news_sm' carregado.")
except Exception as e:
    logging.error(f"Erro ao carregar modelo spaCy: {e}", exc_info=True)
    print(json.dumps({"erro": "Erro ao inicializar spaCy.", "detalhes": str(e)}, ensure_ascii=False)) # Erro JSON para Java (stdout)
    print(f"Erro fatal ao carregar modelo spaCy: {e}", file=sys.stderr)
    sys.exit(1)
# --------------------------------------------

# --- Constantes e Caminhos ---
CAMINHO_DICIONARIO_SINONIMOS = r'C:\Users\MENICIO JR\Desktop\Programa_heber2- antes\src\main\resources\resultado_similaridade.txt'
CAMINHO_PERGUNTAS_INTERESSE = r'C:\Users\MENICIO JR\Desktop\Programa_heber2- antes\src\main\resources\perguntas_de_interesse.txt'
CAMINHO_DICIONARIO_VERBOS = r'C:\Users\MENICIO JR\Desktop\Programa_heber2- antes\src\main\resources\dicionario_verbos.txt'
# --------------------------------------------------------------------

pronomes_interrogativos = ['quem', 'o que', 'que', 'qual', 'quais', 'quanto', 'quantos', 'onde', 'como', 'quando', 'por que', 'porquê']

# --- Funções Auxiliares ---

def carregar_arquivo_linhas(caminho):
    """Carrega linhas de um arquivo, com detecção de encoding e tratamento de erros."""
    logging.debug(f"Tentando carregar arquivo: {caminho}")
    try:
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252']
        content = None
        for enc in encodings_to_try:
            try:
                with open(caminho, 'r', encoding=enc) as f:
                    content = f.read()
                logging.info(f"Arquivo {caminho} lido com encoding: {enc}")
                break
            except UnicodeDecodeError:
                logging.warning(f"Falha ao ler {caminho} com encoding {enc}.")
                continue
            except FileNotFoundError: raise
            except Exception as inner_e:
                 logging.warning(f"Erro inesperado ao tentar ler {caminho} com {enc}: {inner_e}")
                 continue

        if content is None: raise IOError(f"Não foi possível ler {caminho} com {encodings_to_try}")
        linhas = [linha.strip() for linha in content.splitlines() if linha.strip()]
        if not linhas: logging.warning(f"Arquivo {caminho} vazio ou sem conteúdo útil.")
        logging.debug(f"Carregadas {len(linhas)} linhas de {caminho}")
        return linhas
    except FileNotFoundError:
        logging.error(f"Arquivo essencial não encontrado: {caminho}", exc_info=True)
        print(json.dumps({"erro": f"Configuração '{os.path.basename(caminho)}' não encontrada."}, ensure_ascii=False)) # Erro JSON para Java (stdout)
        print(f"Erro fatal: Arquivo não encontrado: {caminho}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.error(f"Erro ao ler ou processar arquivo {caminho}: {e}", exc_info=True)
        print(json.dumps({"erro": f"Erro ao ler configuração: {os.path.basename(caminho)}."}, ensure_ascii=False)) # Erro JSON para Java (stdout)
        print(f"Erro fatal ao ler arquivo {caminho}: {e}", file=sys.stderr)
        sys.exit(1)

def carregar_dicionario_sinonimos(caminho):
    """Carrega dicionário: chave=[(sinonimo, valor)]."""
    dicionario = {}
    linhas = carregar_arquivo_linhas(caminho)
    try:
        linha_num = 0
        # Regex permite letras, números e underscore na chave
        chave_regex = re.compile(r"([\w_]+)\s*=\s*\[(.*)\]")
        for linha in linhas:
            linha_num += 1
            linha_norm = linha.lower().replace('�', '?') # Corrige '?' antes
            match = chave_regex.match(linha_norm)
            if match:
                chave = match.group(1)
                valores_str = match.group(2)
                valores = []
                # Regex para extrair ('sinonimo', valor_float)
                for sin_match in re.finditer(r"'([^']+)'\s*,\s*([\d.]+)", valores_str):
                     sinonimo = sin_match.group(1).strip()
                     try:
                         valor = float(sin_match.group(2))
                         valores.append((sinonimo, valor))
                     except ValueError:
                         logging.warning(f"Ignorando valor float inválido para '{sinonimo}' (chave '{chave}') em {caminho}, linha {linha_num}")
                if valores:
                    dicionario[chave] = valores
                else:
                    # Log se a lista de valores estiver vazia após o parse, mesmo que a chave exista
                    logging.warning(f"Nenhum par (sinônimo, valor) válido encontrado para chave '{chave}' em {caminho}, linha {linha_num}. Linha original: {linha[:50]}...")
            elif linha and not linha.startswith('#'):
                 logging.warning(f"Formato inválido ignorado no dicionário {os.path.basename(caminho)}, linha {linha_num}: {linha[:100]}...")
    except Exception as e:
        logging.error(f"Erro fatal ao parsear dicionário {caminho}: {e}", exc_info=True)
        print(json.dumps({"erro": "Erro no formato interno do dicionário de sinônimos."}, ensure_ascii=False)) # Erro JSON para Java (stdout)
        print(f"Erro fatal ao parsear dicionário de sinônimos {caminho}: {e}", file=sys.stderr)
        sys.exit(1)
    logging.info(f"Dicionário '{os.path.basename(caminho)}' carregado: {len(dicionario)} chaves.")
    return dicionario

def normalizar_texto(texto):
    """Normaliza texto: minúsculas, remove acentos, trata caracteres problemáticos."""
    if not texto: return ""
    texto = str(texto).strip().lower()
    replacements = {'�': 'ç', '�': 'ã', '�': 'õ', '�': 'á', '�': 'é', '�': 'í', '�': 'ó', '�': 'ú',
                    '�': 'â', '�': 'ê', '�': 'ô', '�': 'à', '?':' '} # Substitui '?' por espaço? ou remove?
    for problematic, replacement in replacements.items():
         texto = texto.replace(problematic, replacement)
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    except Exception as e:
        logging.warning(f"Erro ao normalizar texto Unicode: '{texto[:50]}...'. Erro: {e}")
        return texto

def extrair_entidades_spacy(texto):
    """Extrai SPO e NER."""
    elementos = {"sujeito": [], "predicado": [], "objeto": []}
    entidades_nomeadas = {}
    try:
        doc = nlp(texto)
        processed_tokens = set() # Para evitar adicionar subfrases múltiplas vezes

        for token in doc:
            # Sujeito (nominal)
            if token.dep_ == "nsubj" and token.i not in processed_tokens and token.text.lower() not in pronomes_interrogativos :
                sub_phrase = " ".join(t.text for t in token.subtree).strip()
                elementos["sujeito"].append(sub_phrase)
                processed_tokens.update(t.i for t in token.subtree)
            # Predicado (verbo principal)
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "conj"):
                elementos["predicado"].append(token.lemma_)
            # Objeto ou Complemento Obliquio
            if token.dep_ in {"obj", "obl", "dobj", "pobj", "iobj"} and token.i not in processed_tokens and token.text.lower() not in pronomes_interrogativos:
                 phrase = "".join(t.text_with_ws for t in token.subtree).strip()
                 # Limpeza adicional (remover 'em dd/mm/yyyy' do final se for data)
                 data_match = re.search(r'\s+em\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}$', phrase, re.IGNORECASE)
                 if data_match:
                      phrase = phrase[:data_match.start()].strip()

                 if phrase and len(phrase) > 1:
                      elementos["objeto"].append(phrase)
                 processed_tokens.update(t.i for t in token.subtree)

        # Limpeza e Deduplicação SPO
        for key in elementos:
             elementos[key] = sorted(list(set(el for el in elementos[key] if el)), key=lambda x: texto.find(x))

        # Extração de Entidades Nomeadas (NER)
        for ent in doc.ents:
            label = ent.label_
            text = ent.text.strip().replace('.', '')
            if text and len(text) > 1 and not text.isdigit():
                 # Verifica se a entidade NER já não está contida no objeto/sujeito extraído
                 already_in_spo = False
                 for spo_list in elementos.values():
                      for item in spo_list:
                           if text in item:
                                already_in_spo = True
                                break
                      if already_in_spo: break
                 # Adiciona apenas se não fizer parte de SPO maior (evita redundância)
                 #if not already_in_spo: # Comentado para sempre adicionar NER por enquanto
                 entidades_nomeadas.setdefault(label, []).append(text)


        logging.debug(f"Texto para spaCy: '{texto}'")
        logging.debug(f"Elementos extraídos (SPO): {elementos}")
        logging.debug(f"Entidades Nomeadas extraídas (NER): {entidades_nomeadas}")
    except Exception as e:
        logging.error(f"Erro durante extração spaCy: {e}", exc_info=True)
        print(f"Erro durante extração spaCy: {e}", file=sys.stderr)
        elementos = {"sujeito": [], "predicado": [], "objeto": []}; entidades_nomeadas = {}
    return elementos, entidades_nomeadas

def extrair_data(texto):
    """Extrai data (dd/mm/yyyy, dd-mm-yyyy, hoje, ontem) e normaliza para AAAA-MM-DD."""
    padroes = [ r'\b(\d{1,2}/\d{1,2}/\d{4})\b', r'\b(\d{1,2}-\d{1,2}-\d{4})\b', ]
    texto_lower = texto.lower()
    data_encontrada_str = None
    for padrao in padroes:
        match = re.search(padrao, texto)
        if match: data_encontrada_str = match.group(1); break
    if data_encontrada_str:
        formats_to_try = ["%d/%m/%Y", "%d-%m-%Y"]
        for fmt in formats_to_try:
            try:
                data_normalizada = datetime.strptime(data_encontrada_str, fmt).strftime("%Y-%m-%d")
                logging.info(f"Data explícita '{data_encontrada_str}' normalizada para: {data_normalizada}")
                return data_normalizada
            except ValueError: continue
        logging.warning(f"Data '{data_encontrada_str}' encontrada, formato não reconhecido.")
        return data_encontrada_str
    if "hoje" in texto_lower: data_normalizada = datetime.now().strftime("%Y-%m-%d"); logging.info("Data 'hoje' normalizada."); return data_normalizada
    if "ontem" in texto_lower: data_normalizada = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"); logging.info("Data 'ontem' normalizada."); return data_normalizada
    logging.info("Nenhuma data encontrada.")
    return None

def encontrar_termo_dicionario(frase, dicionario, limiar=0.70):
    """Encontra a melhor chave no dicionário de sinônimos para uma frase."""
    if not dicionario or not frase: logging.debug(f"Dicio ou frase vazia: '{frase}'"); return None
    frase_norm = normalizar_texto(frase)
    if not frase_norm: logging.debug("Frase normalizada vazia."); return None
    melhor_chave = None
    maior_similaridade = limiar - 0.01
    logging.debug(f"Buscando termo para frase normalizada: '{frase_norm}' (Limiar: {limiar})")
    for chave, sinonimos in dicionario.items():
        chave_norm = normalizar_texto(chave)
        # Log muito verboso, talvez desativar em produção
        # logging.log(5, f"  Comparando com chave '{chave_norm}'...")
        # --- Comparação com chave principal ---
        sim_chave = difflib.SequenceMatcher(None, frase_norm, chave_norm).ratio()
        if sim_chave >= maior_similaridade:
            if sim_chave > maior_similaridade or (sim_chave == maior_similaridade and len(chave) < len(melhor_chave or "z"*100)): # Prefere mais curto em empate
                maior_similaridade = sim_chave
                melhor_chave = chave
                logging.debug(f"  -> Novo melhor (chave): '{melhor_chave}' similaridade {maior_similaridade:.4f} >= {limiar}")
        # --- Comparação com sinônimos ---
        for sinonimo, _ in sinonimos:
            sinonimo_norm = normalizar_texto(sinonimo)
            if not sinonimo_norm: continue # Pula sinônimos que viram vazios após normalização
            # logging.log(5, f"    Comparando com sinônimo '{sinonimo_norm}'...")
            sim_sin = difflib.SequenceMatcher(None, frase_norm, sinonimo_norm).ratio()
            if sim_sin >= maior_similaridade:
                 if sim_sin > maior_similaridade or (sim_sin == maior_similaridade and len(chave) < len(melhor_chave or "z"*100)):
                     maior_similaridade = sim_sin
                     melhor_chave = chave # Associa à chave principal
                     logging.debug(f"  -> Novo melhor (sinônimo '{sinonimo_norm}'): chave '{melhor_chave}' similaridade {maior_similaridade:.4f} >= {limiar}")
    if melhor_chave:
        logging.info(f"Melhor termo encontrado para '{frase}': '{melhor_chave}' (Similaridade: {maior_similaridade:.2f})")
    else:
        logging.info(f"Nenhum termo encontrado para '{frase}' com similaridade >= {limiar}")
    return melhor_chave

# --- Função encontrar_pergunta_similar (VERSÃO CONFIRMADA - Retorna 3 valores) ---
def encontrar_pergunta_similar(pergunta_usuario, templates_linhas, limiar=0.70): # Limiar pode ser ajustado
    """
    Encontra a linha mais similar no arquivo de templates e extrai o NOME do template.
    Retorna (nome_template, similaridade, linha_completa_original) ou (None, 0, None).
    Assume formato: "Nome do Template - Pergunta exemplo..."
    """
    maior_similaridade = 0.0
    template_nome_final = None
    linha_template_correspondente = None
    pergunta_usuario_norm = normalizar_texto(pergunta_usuario)
    logging.debug(f"Normalizada pergunta para template: '{pergunta_usuario_norm}' (Limiar: {limiar})")

    if not templates_linhas:
         logging.error("Lista de templates de perguntas vazia."); print("Erro...", file=sys.stderr); return None, 0, None

    logging.debug(f"Comparando com {len(templates_linhas)} templates.")
    for i, linha_template in enumerate(templates_linhas):
        try:
            logging.debug(f"  Linha {i+1}: '{linha_template}'")
            if " - " in linha_template:
                partes = linha_template.split(" - ", 1)
                nome_template_atual = partes[0].strip()
                pergunta_exemplo = partes[1].strip()
            else:
                 logging.warning(f"    Formato inválido (sem ' - '): '{linha_template}'. Ignorando.")
                 continue

            template_norm = normalizar_texto(pergunta_exemplo)
            similaridade = difflib.SequenceMatcher(None, pergunta_usuario_norm, template_norm).ratio()
            logging.debug(f"    -> Exemplo norm: '{template_norm}', Similaridade: {similaridade:.4f}")

            if similaridade > maior_similaridade:
                logging.debug(f"    -> NOVA MAIOR SIMILARIDADE! Nome: '{nome_template_atual}'")
                maior_similaridade = similaridade
                template_nome_final = nome_template_atual
                linha_template_correspondente = linha_template

        except Exception as e:
            logging.exception(f"Erro ao processar linha template {i+1}: '{linha_template}'")
            print(f"Erro linha template {i+1}: {e}", file=sys.stderr)
            continue

    if maior_similaridade < limiar:
         logging.warning(f"Similaridade máxima ({maior_similaridade:.2f}) abaixo do limiar ({limiar}).")
         template_nome_final, maior_similaridade, linha_template_correspondente = None, 0, None

    logging.info(f"!!! RESULTADO encontrar_pergunta_similar: nome='{template_nome_final}', similaridade={maior_similaridade:.4f}")
    return template_nome_final, maior_similaridade, linha_template_correspondente
# --- FIM encontrar_pergunta_similar ---

def mapear_para_placeholders(pergunta_usuario_original, elementos, entidades_nomeadas, data, dicionario_sinonimos):
    """Mapeia para placeholders semânticos."""
    mapeamentos = {}
    logging.debug("Iniciando mapeamento semântico...")
    if data: mapeamentos['#DATA'] = data; logging.debug(f"Mapeado #DATA: {data}")

    entidade_principal_str = None
    tipo_entidade_principal = None
    # Tenta ORG primeiro, depois GPE (com validação de ticker)
    if 'ORG' in entidades_nomeadas and entidades_nomeadas['ORG']:
        entidade_principal_str = entidades_nomeadas['ORG'][0]
        tipo_entidade_principal = "ORG"
    elif 'GPE' in entidades_nomeadas and entidades_nomeadas['GPE']:
         possible_ticker = entidades_nomeadas['GPE'][0]
         if re.match(r"^[A-Z]{4}\d{1,2}$", possible_ticker.upper()):
              entidade_principal_str = possible_ticker
              tipo_entidade_principal = "GPE (Ticker?)"

    if entidade_principal_str:
        entidade_principal_str = entidade_principal_str.replace('.', '').strip().upper()
        mapeamentos['#ENTIDADE'] = entidade_principal_str
        logging.info(f"Mapeado #ENTIDADE: '{entidade_principal_str}' (Tipo NER: {tipo_entidade_principal})")
    else:
        logging.warning("Não identificada entidade principal (Empresa/Ticker) via NER.")
        # Tentar extrair do sujeito se não for pronome?
        sujeito_str = " ".join(elementos.get('sujeito', [])).strip()
        if sujeito_str and len(sujeito_str) > 2 and not any(pron in sujeito_str for pron in ['ele', 'ela', 'isso', 'qual']):
             # É um nome plausível? Pode precisar de mais validação.
             logging.info(f"Tentando usar sujeito '{sujeito_str}' como #ENTIDADE (fallback)")
             mapeamentos['#ENTIDADE'] = sujeito_str.upper() # Padroniza
        else:
            logging.error("Falha em obter #ENTIDADE do NER ou do sujeito.")
            # Pode ser um erro fatal dependendo do template

    # Mapear Valor Desejado
    valor_desejado_chave = None
    texto_para_busca = ""
    # Prioriza o sujeito se não for só 'o', 'a', 'os', 'as'
    sujeito_str = " ".join(elementos.get('sujeito', [])).strip()
    if sujeito_str and not re.match(r"^(o|a|os|as)$", sujeito_str.lower()):
        texto_para_busca = sujeito_str
    else: # Senão, tenta o objeto
        texto_para_busca = " ".join(elementos.get('objeto', [])).strip()

    # Remove a entidade já encontrada e a data para focar no valor
    if entidade_principal_str: texto_para_busca = texto_para_busca.replace(entidade_principal_str.lower(), "").strip()
    if data: texto_para_busca = texto_para_busca.replace(data, "").strip() # Usa data original para remover
    # Remove preposições e artigos comuns do início/fim
    texto_para_busca = re.sub(r"^(da|de|do|em|na|no|para)\s+", "", texto_para_busca).strip()
    texto_para_busca = re.sub(r"\s+(da|de|do|em|na|no)$", "", texto_para_busca).strip()

    logging.debug(f"Texto final usado para buscar #VALOR_DESEJADO: '{texto_para_busca}'")
    if texto_para_busca:
         valor_desejado_chave = encontrar_termo_dicionario(texto_para_busca, dicionario_sinonimos)

    if valor_desejado_chave:
        mapeamentos['#VALOR_DESEJADO'] = valor_desejado_chave
        logging.info(f"Mapeado #VALOR_DESEJADO para chave dicionário: '{valor_desejado_chave}'")
    else:
        logging.warning(f"Não foi possível mapear #VALOR_DESEJADO para '{texto_para_busca}'.")


    # Mapear Tipo de Ação
    texto_completo_norm = normalizar_texto(pergunta_usuario_original)
    if "ordinaria" in texto_completo_norm or " on " in texto_completo_norm: mapeamentos['#TIPO_ACAO'] = "ORDINARIA"; logging.debug("Map.#TIPO_ACAO: ORDINARIA")
    elif "preferencial" in texto_completo_norm or " pn " in texto_completo_norm: mapeamentos['#TIPO_ACAO'] = "PREFERENCIAL"; logging.debug("Map.#TIPO_ACAO: PREFERENCIAL")

    logging.info(f"Mapeamentos semânticos finais: {mapeamentos}")
    return mapeamentos


# --- Bloco Principal ---
if __name__ == "__main__":
    _JSON_ERROR_PRINTED = False
    logging.info(f"--- Iniciando processamento PLN (PID: {os.getpid()}) ---")

    # 1. Obter Pergunta
    if len(sys.argv) > 1:
        pergunta_usuario = sys.argv[1]
        logging.info(f"Pergunta recebida: '{pergunta_usuario}'")
    else:
        logging.error("Nenhuma pergunta fornecida."); print(json.dumps({"erro":"?"})); sys.exit(1)

    # 2. Carregar Dados
    templates_linhas_interesse = carregar_arquivo_linhas(CAMINHO_PERGUNTAS_INTERESSE)
    dicionario_sinonimos = carregar_dicionario_sinonimos(CAMINHO_DICIONARIO_SINONIMOS)

    # 3. Processamento NLP
    try:
        elementos, entidades_nomeadas = extrair_entidades_spacy(pergunta_usuario)
        data_extraida = extrair_data(pergunta_usuario)
    except Exception as e:
        logging.error("Erro crítico NLP", exc_info=True); print(json.dumps({"erro":"NLP"})); sys.exit(1)

    # 4. Encontrar Template Similar
    template_nome, similaridade, linha_original_template = encontrar_pergunta_similar(pergunta_usuario, templates_linhas_interesse, limiar=0.65)

    if not template_nome:
        logging.error(f"Template não encontrado (Similaridade max: {similaridade:.2f})")
        print(json.dumps({"erro": "Pergunta não compreendida (template).", "similaridade": similaridade}), ensure_ascii=False)
        print("Erro: Template similar não encontrado.", file=sys.stderr); sys.exit(1)

    # 5. Mapear Entidades para Placeholders Semânticos
    mapeamentos_semanticos = mapear_para_placeholders(pergunta_usuario, elementos, entidades_nomeadas, data_extraida, dicionario_sinonimos)

    # Validação Mínima: Precisa de entidade OU valor desejado?
    if not mapeamentos_semanticos.get('#ENTIDADE') and not mapeamentos_semanticos.get('#VALOR_DESEJADO'):
         logging.error("Mapeamento incompleto (sem #ENTIDADE ou #VALOR_DESEJADO)")
         print(json.dumps({"erro": "Não foi possível identificar a empresa/ticker ou o que você deseja saber."}, ensure_ascii=False))
         print("Erro: Mapeamento incompleto.", file=sys.stderr); sys.exit(1)

    # 6. Construir Resposta JSON Final
    resposta_final = {
        "template_nome": template_nome, # Nome curto: "Template XA"
        "mapeamentos": mapeamentos_semanticos,
        "_debug_info": { # Informações úteis para depuração no Java
             "elementos_extraidos": elementos,
             "entidades_nomeadas": entidades_nomeadas,
             "data_extraida": data_extraida,
             "similaridade_template": round(similaridade, 4),
             "linha_template_correspondente": linha_original_template
        }
    }

    # Log ANTES de imprimir
    logging.debug(f"JSON final para stdout: {json.dumps(resposta_final, ensure_ascii=False, indent=2)}")

    # --- IMPRESSÃO FINAL PARA STDOUT ---
    print(json.dumps(resposta_final, ensure_ascii=False, indent=2))
    logging.info("--- Processamento PLN concluído ---")
    # -----------------------------------
