# win-rl

Agente de reinforcement learning para scalp/day trade no mini índice (WIN), 1 contrato.
Tudo aqui é **leitura de dados + simulação**. Nenhum script envia ordem ao MT5, e os
scripts que conectam no MT5 abortam se a conta não for DEMO.

## Megazord (o jeito principal de usar)

1. Edite o **`megazord.ini`** (`MODO`, `CONECTAR_MT5` etc.).
2. Rode o **`run.bat`**. O escritório abre no navegador (`http://localhost:8765/escritorio.html`), e o painel com gráficos fica em `/time.html`.

| MODO | O que acontece |
|---|---|
| `treino` | Os 5 traders fixos (`team/trader_1..5`) continuam estudando, rodada após rodada. Ao fim de cada rodada, cada um faz uma prova na validação, e o aprendiz só vira **campeão** (o que vota) se for melhor que o campeão anterior. Depois, o time faz uma prova em pregões nunca estudados. Ctrl+C para parar; o aprendizado fica salvo em `team/`. |
| `replay` | Os campeões jogam os últimos `DIAS_TESTE_PAINEL` pregões de teste, para assistir. |
| `ao_vivo` | Os campeões votam a cada candle de M1 fechado. `CONECTAR_MT5 = true` lê do MT5 (só conta demo); `false` simula o último pregão no relógio. Com `ENVIAR_ORDENS = false`, só mostra no painel. |

### Ordens no MT5 (`ENVIAR_ORDENS = true`, só conta demo)

O Megazord manda ordens a mercado de 1 contrato no WIN vigente (ex.: `WINZ26`), seguindo a posição decidida pela votação, com as mesmas regras do treino. Para ligar: `MODO = ao_vivo`, `CONECTAR_MT5 = true`, `ENVIAR_ORDENS = true` e o botão **Algo Trading** do MT5 ligado.

- Recusa qualquer conta que não seja **demo** (`winrl/broker.py`).
- Stop de emergência no servidor em cada entrada (`STOP_EMERGENCIA_PTS`).
- Perda máxima do dia em R$ e número máximo de ordens: atingiu, zera e para de entrar.
- Zera `ZERAR_ANTES_MIN` minutos antes do último candle e ao fechar com Ctrl+C.
- Criar um arquivo chamado `PARAR` (sem extensão) na pasta do projeto zera e bloqueia novas entradas.
- Nunca entra no meio de um trade que o Megazord começou antes de o programa ligar.
- Cada ordem fica registrada em `logs\ordens_AAAA-MM-DD.csv`.
- Teste rápido (compra 1 e zera): `.venv\Scripts\python scripts\teste_ordem.py`.

Promoção: na prova de cada rodada, aprendiz e campeão jogam os mesmos pregões de validação e o resultado é comparado **dia a dia** (teste t pareado). O aprendiz só vira campeão se for melhor na média com `t >= PROMOCAO_T_MIN`. Cada trader estuda com uma taxa de aprendizado diferente e, depois de `RESET_APRENDIZ_APOS` rodadas sem promoção, o aprendiz recomeça do campeão com passos menores. O `history.json` de cada trader guarda o `t_stat`, os dias melhores e piores e se houve recomeço.

Regra de votação: o Megazord **entra** quando `VOTOS_PARA_ENTRAR` traders querem a mesma direção, **mantém** enquanto pelo menos `VOTOS_PARA_MANTER` ainda querem, **vira a mão** se `VOTOS_PARA_ENTRAR` querem o lado oposto e, fora isso, zera. Cada trader também opera sua própria "conta-sombra", e é ela que aparece no ranking.

Na primeira vez, se existirem os modelos `models/pen5y_WIN_N_s0..4`, o time nasce a partir deles; senão, começa do zero. Para recomeçar o time, apague a pasta `team/`.

Para testar sem mexer no time de verdade: `.venv\Scripts\python scripts\run.py tests\teste_treino.ini` (também existem `teste_replay.ini`, `teste_vivo.ini` e `teste_mt5.ini`, que usam a pasta `team_teste/` e a porta 8766).

## Estrutura

| Caminho | O quê |
|---|---|
| `winrl/features.py` | Features do agente: o mesmo código para treino, backtest e (futuro) tempo real |
| `winrl/env.py` | Ambiente Gymnasium: 1 episódio = 1 pregão, decide no fechamento e executa na abertura seguinte, custos, zeragem no fim do dia |
| `winrl/backtest.py` | Simulador de regras fixas (SL/TP pessimista dentro do candle) + métricas |
| `winrl/strategies.py` | Baselines: aleatório e rompimento da faixa de abertura |
| `winrl/synthetic.py` | Pregões sintéticos (`random` = nada a aprender, `momentum` = padrão real) |
| `winrl/data.py` | Carrega o formato canônico e separa treino/validação/teste **por data** |
| `scripts/inventory.py` | Inventário do que a corretora entrega (histórico, ticks, spread por hora) |
| `scripts/export_rates.py` | MT5 → `data/<SIMBOLO>/<TF>.parquet` |
| `scripts/train.py` | Treino PPO, checkpoints versionados, `best_model` escolhido na validação |
| `scripts/evaluate.py` | Avaliação congelada contra aleatório com a mesma frequência e "não operar" |
| `scripts/replay.py` | Gráfico de replay (candles, entradas/saídas, SL/TP, volume, PnL) |

## Uso

```
.venv\Scripts\python scripts\export_rates.py WIN$N
.venv\Scripts\python scripts\check_env.py
.venv\Scripts\python scripts\train.py --data momentum --steps 800000      # sanidade
.venv\Scripts\python scripts\train.py --data WIN$N --steps 2000000 --seed 0
.venv\Scripts\python scripts\train.py --data WIN$N --steps 1000000 --seed 0 --resume models\WIN_N_s0\ckpt_1000000_steps.zip
.venv\Scripts\python scripts\evaluate.py --data WIN$N --model models\WIN_N_s0\best_model.zip --split test
.venv\Scripts\python scripts\replay.py --trades data\trades_WIN_N_s0_test.parquet --day 2026-10-15
```

## Painel ao vivo (`dashboard/`)

- **`assistir_treino.bat`**: treina do zero e, a cada 20 mil passos, a política atual joga um pregão
  da validação no painel, junto com as curvas de aprendizado (reward, entropia, trades).
- **`assistir_modelo.bat [pasta_do_modelo] [train|val|test]`**: um modelo já treinado joga todos os
  pregões do período, candle a candle.

Os dois abrem `http://localhost:8765`, que é servido por `scripts/serve.py` e só fica acessível nesta
máquina. Atalhos: espaço para tocar/pausar, ← → para andar um candle. Clicar numa barra de
"Pregões publicados" abre aquele dia.

## Custos assumidos (em `winrl/backtest.py`, classe `Costs`)
Meio spread 3,25 pts (spread médio medido nos ticks ≈ 6,5) + taxas ≈ 1,25 pts por lado
= **9 pts por ida e volta**. As taxas são estimativa: confira com a tabela da sua corretora/B3.

## Limitações conhecidas
- Só ~9 meses de M1 (limite "Máx. barras no gráfico" = 100000 no MT5). Pôr em Ilimitado e reexportar.
- Execução sempre a mercado; sem modelagem de fila de ordem limitada.
- SL/TP no `backtest.py` resolvidos de forma pessimista no M1; com ticks dá para resolver de verdade.
- Ainda não existem os modos `paper`/`live`.
