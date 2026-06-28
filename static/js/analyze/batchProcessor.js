/**
 * batchProcessor.js
 *
 * Processes unassigned transactions in batches of 10 via the analyze API.
 */

const BATCH_SIZE = 10;

export class BatchProcessor {
    constructor(fileId) {
        this.fileId = fileId;
        this.running = false;
        this.offset = 0;
    }

    bindUI() {
        this.progressBar = document.getElementById('batchProgressBar');
        this.progressText = document.getElementById('batchProgressText');
        this.statusText = document.getElementById('batchStatusText');
        this.startButton = document.getElementById('startBatchProcess');
        this.stopButton = document.getElementById('stopBatchProcess');

        if (this.startButton) {
            this.startButton.addEventListener('click', () => this.start());
        }
        if (this.stopButton) {
            this.stopButton.addEventListener('click', () => this.stop());
        }
    }

    async start() {
        if (this.running) {
            return;
        }

        this.running = true;
        this.offset = 0;
        this.setUiRunning(true);
        this.updateStatus('Starting batch processing...');

        try {
            while (this.running) {
                const result = await this.processNextBatch();
                if (!result || !result.has_more) {
                    this.updateStatus('Batch processing complete. Review suggestions on this page.');
                    break;
                }
                this.offset = result.next_offset;
            }
        } catch (error) {
            console.error('Batch processing error:', error);
            this.updateStatus(`Error: ${error.message}`);
        } finally {
            this.running = false;
            this.setUiRunning(false);
        }
    }

    stop() {
        this.running = false;
        this.updateStatus('Batch processing stopped.');
        this.setUiRunning(false);
    }

    async processNextBatch() {
        this.updateStatus(`Processing next ${BATCH_SIZE} transactions...`);

        const response = await fetch(`/api/analyze/${this.fileId}/process-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                offset: this.offset,
                batch_size: BATCH_SIZE,
            }),
        });

        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.error || `Request failed (${response.status})`);
        }

        const result = await response.json();
        this.applyBatchResults(result.results || []);
        this.updateProgress(result);

        if (result.processed > 0) {
            this.updateStatus(
                `Processed ${result.processed} transaction(s). ` +
                `${result.remaining} remaining.`
            );
        } else if (result.total_unprocessed === 0) {
            this.updateStatus('All transactions already have accounts or explanations.');
        }

        return result;
    }

    applyBatchResults(results) {
        results.forEach((item) => {
            const select = document.querySelector(`select[name="account_${item.transaction_id}"]`);
            if (!select) {
                return;
            }

            if (item.applied_account_id) {
                select.value = String(item.applied_account_id);
                select.classList.add('border-success');
                return;
            }

            const suggestion = item.suggestion || {};
            if (suggestion.success && suggestion.account) {
                const match = Array.from(select.options).find((option) =>
                    option.text.toLowerCase().includes(suggestion.account.toLowerCase())
                );
                if (match) {
                    select.value = match.value;
                    select.classList.add('border-info');
                }
            }
        });
    }

    updateProgress(result) {
        const total = result.total_unprocessed || 0;
        const done = Math.max(0, total - (result.remaining || 0));
        const percent = total > 0 ? Math.round((done / total) * 100) : 100;

        if (this.progressBar) {
            this.progressBar.style.width = `${percent}%`;
            this.progressBar.textContent = `${percent}%`;
        }
        if (this.progressText) {
            this.progressText.textContent = `${done} / ${total} reviewed`;
        }
    }

    updateStatus(message) {
        if (this.statusText) {
            this.statusText.textContent = message;
        }
    }

    setUiRunning(isRunning) {
        if (this.startButton) {
            this.startButton.disabled = isRunning;
        }
        if (this.stopButton) {
            this.stopButton.disabled = !isRunning;
        }
    }
}

export default BatchProcessor;
