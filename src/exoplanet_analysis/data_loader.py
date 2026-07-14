import os
import lightkurve as lk
from lightkurve import search_targetpixelfile, TessTargetPixelFile

def download_tess_tpf(target_id, sector=None, download_dir='./data/raw'):
    """
    Busca e baixa o Target Pixel File do TESS para um alvo
    
    Parametros:
    target_id : str
        Nome ou ID do alvo
    sector : int, opcional
        Setor especifico do TESS a ser baixado, se None, baixa todos disponiveis
    download_dir : str
        Diretorio onde os arquivos .fits serao salvos
        
    Retorno:
    tpf : lightkurve.TessTargetPixelFile
        Objeto TPF carregado na memoria e salvo no disco
    """
    
    # 1. Realiza a busca no MAST
    search_result = search_targetpixelfile(target_id, mission='TESS', sector=sector)
    
    if len(search_result) == 0:
        raise ValueError(f"Nenhum dado encontrado no MAST para o alvo {target_id} no setor {sector}.")
    
    
    # A quality_bitmask='default' já remove erros conhecidos e dados ruins, podendo ser ajustados conforme necessidade
    tpf = search_result.download(download_dir=download_dir, quality_bitmask='default')
    
    if tpf is None:
        raise ConnectionError("Falha ao baixar o arquivo. Verifique sua conexão com o MAST")
        
    return tpf

def load_tpf_from_disk(filepath):
    """
    Le um arquivo TPF (.fits) que ja esta salvo no computador
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arquivo nao encontrado: {filepath}")
        
    try:
        tpf = TessTargetPixelFile(filepath)
        return tpf
    except Exception as e:
        raise RuntimeError(f"Erro ao ler o arquivo FITS {filepath}: {e}")