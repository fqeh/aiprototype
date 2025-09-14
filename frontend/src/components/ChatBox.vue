<template>
  <v-app>
    <!-- Fixed App Bar -->
    <v-app-bar flat class="appbar outlined-2 r6 fixed-bar" height="64">
      <v-toolbar-title class="fw-700">Chat Playground</v-toolbar-title>
      <v-spacer />
      <v-text-field
        :model-value="`User: ${userId}`"
        readonly
        density="comfortable"
        variant="outlined"
        hide-details
        class="outlined-2 r6 user-box"
      />
    </v-app-bar>

    <v-main class="app-bg main-offset">
      <v-container class="py-6" style="max-width: 900px;">
        <v-row class="mb-4" align="center">
          <v-col cols="12" md="6">
            <v-select
              label="Select LLM"
              :items="llmOptions"
              item-title="label"
              item-value="value"
              v-model="pendingModel"
              variant="outlined"
              class="outlined-2 r6"
              :loading="loadingModels"
              :disabled="loadingModels || busy"
              @update:model-value="onModelPick"
            >
              <template #append-inner>
                <FontAwesomeIcon icon="fa-solid fa-chevron-down" />
              </template>
            </v-select>
          </v-col>

          <v-col cols="12" md="6" class="text-right">
            <v-btn
              variant="outlined"
              color="error"
              class="outlined-2 r6 fw-700 center-btn"
              :disabled="busy"
              @click="confirmReset = true"
            >
              <FontAwesomeIcon icon="fa-solid fa-rotate-left" class="mr-1" />
              Reset Chat
            </v-btn>
          </v-col>
        </v-row>

        <!-- Chat + Input -->
        <v-card variant="outlined" class="outlined-2 r6">
          <v-card-title class="fw-700">Conversation ({{ modelLabel }})</v-card-title>
          <v-divider />

          <v-card-text>
            <div class="chat-scroll r6 outlined-2-soft" ref="scrollEl">
              <div
                v-for="(m, i) in visibleMessages"
                :key="i"
                class="mb-3 d-flex bubble-row"
                :class="m.role === 'user' ? 'justify-end' : 'justify-start'"
              >
                <div class="bubble-wrap" :class="m.role === 'user' ? 'user-side' : 'assistant-side'">
                  <!-- Bubble (normal) -->
                  <v-sheet
                    v-if="!m.editing"
                    class="pa-3 bubble outlined-2 r6"
                    :class="m.role === 'user' ? 'bubble-user' : 'bubble-assistant'"
                    max-width="80%"
                    variant="outlined"
                  >
                    <div class="mb-1" v-if="m.role !== 'system'">
                      <strong>{{ m.role === 'user' ? 'You' : modelLabel }}</strong>
                    </div>
                    <div style="white-space: pre-wrap;">
                      {{ m.content }}
                    </div>

                    <!-- Icon-only actions -->
                    <div class="bubble-actions">
                      <!-- Copy (both user + assistant) -->
                      <v-btn
                        icon
                        variant="text"
                        density="comfortable"
                        class="icon-dark"
                        @click="copyMessage(m.content)"
                        :aria-label="`Copy message`"
                      >
                        <FontAwesomeIcon icon="fa-regular fa-copy" />
                      </v-btn>

                      <!-- Edit (user only) -->
                      <v-btn
                        v-if="m.role === 'user'"
                        icon
                        variant="text"
                        density="comfortable"
                        class="icon-dark ml-1"
                        @click="startEdit(i)"
                        :aria-label="`Edit message`"
                      >
                        <FontAwesomeIcon icon="fa-regular fa-pen-to-square" />
                      </v-btn>
                    </div>
                  </v-sheet>

                  <!-- Bubble (edit mode) -->
                  <v-sheet
                    v-else
                    class="pa-3 bubble outlined-2 r6 bubble-user"
                    max-width="80%"
                    variant="outlined"
                  >
                    <div class="text-caption mb-1"><strong>Edit your message</strong></div>
                    <v-textarea
                      v-model="editBuffer"
                      auto-grow
                      rows="2"
                      variant="outlined"
                      class="outlined-2 r6"
                      :disabled="busy"
                    />
                    <div class="edit-actions mt-2">
                      <v-btn
                        variant="outlined"
                        class="outlined-2 r6 fw-700 center-btn"
                        size="small"
                        :disabled="busy"
                        :loading="busy"
                        @click="confirmEdit(i)"
                      >
                        <FontAwesomeIcon icon="fa-solid fa-check" class="mr-1" />
                        OK
                      </v-btn>

                      <v-btn
                        variant="outlined"
                        class="outlined-2 r6 fw-700 center-btn"
                        size="small"
                        color="error"
                        :disabled="busy"
                        @click="cancelEdit(i)"
                      >
                        <FontAwesomeIcon icon="fa-solid fa-xmark" class="mr-1" />
                        Cancel
                      </v-btn>
                    </div>
                  </v-sheet>
                </div>
              </div>

              <!-- Typing indicator -->
              <div v-if="busy" class="d-flex justify-start mb-3">
                <v-sheet class="pa-3 bubble bubble-assistant outlined-2 r6" variant="outlined">
                  <span>…thinking</span>
                </v-sheet>
              </div>
            </div>
          </v-card-text>

          <!-- Input -->
          <v-divider />
          <v-card-actions class="flex-column">
            <v-textarea
              v-model="draft"
              label="Type your message…"
              auto-grow
              rows="2"
              variant="outlined"
              class="outlined-2 r6 w-100 msg-textarea"
              :disabled="busy"
              @keydown.enter.exact.prevent="sendDraft"
            />
            <div class="d-flex w-100 align-center">
              <v-btn
                :disabled="busy || !draft.trim()"
                :loading="busy"
                variant="outlined"
                class="outlined-2 r6 fw-700 center-btn"
                @click="sendDraft"
              >
                <FontAwesomeIcon icon="fa-solid fa-paper-plane" class="mr-1" />
                Send
              </v-btn>

              <v-btn
                class="ml-2 outlined-2 r6 fw-700 center-btn"
                variant="outlined"
                color="secondary"
                :disabled="busy"
                @click="draft = ''"
              >
                <FontAwesomeIcon icon="fa-solid fa-xmark" class="mr-1" />
                Clear
              </v-btn>

              <v-spacer />
            </div>
          </v-card-actions>
        </v-card>
      </v-container>
    </v-main>

    <!-- Confirm: model change -->
    <v-dialog v-model="confirmModelChange" max-width="460">
      <v-card variant="outlined" class="outlined-2 r6">
        <v-card-title class="fw-700">Restart chat with {{ pendingModelLabel }}?</v-card-title>
        <v-card-text>
          Changing the LLM clears this conversation and starts fresh with
          <strong>{{ pendingModelLabel }}</strong>. Continue?
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="outlined" class="outlined-2 r6 center-btn" @click="cancelModelChange">Cancel</v-btn>
          <v-btn color="primary" variant="outlined" class="outlined-2 r6 fw-700 center-btn" @click="applyModelChange">Restart</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Confirm: manual reset -->
    <v-dialog v-model="confirmReset" max-width="420">
      <v-card variant="outlined" class="outlined-2 r6">
        <v-card-title class="fw-700">Clear conversation?</v-card-title>
        <v-card-text>This removes all messages for this session.</v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="outlined" class="outlined-2 r6 center-btn" @click="confirmReset = false">Cancel</v-btn>
          <v-btn color="error" variant="outlined" class="outlined-2 r6 fw-700 center-btn" @click="resetChat">Clear</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <v-snackbar v-model="snack.open" :timeout="2200" class="outlined-2 r6">
      {{ snack.text }}
    </v-snackbar>
  </v-app>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import axios from 'axios'

/* ── Font Awesome ───────────────────────────────────────────────────── */
import { library } from '@fortawesome/fontawesome-svg-core'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'
import {
  faPaperPlane,
  faRotateLeft,
  faCheck,
  faXmark,
  faChevronDown
} from '@fortawesome/free-solid-svg-icons'
import { faPenToSquare, faCopy } from '@fortawesome/free-regular-svg-icons'
library.add(faPaperPlane, faRotateLeft, faPenToSquare, faCheck, faXmark, faChevronDown, faCopy)

/* ── API base (robust for subpaths + absolute env) ──────────────────── */
function normalizeTrailingSlash(u) {
  return u.replace(/\/+$/, '/') // keep exactly one trailing slash
}
function computeApiBase() {
  const raw = import.meta.env.VITE_API_BASE_URL && String(import.meta.env.VITE_API_BASE_URL).trim()
  // If env is absolute (http/https), use as-is
  if (raw && /^https?:\/\//i.test(raw)) return normalizeTrailingSlash(raw)
  // If env is a relative path, resolve against origin
  if (raw) {
    const rel = raw.replace(/^\/+/, '') // strip leading slashes
    return normalizeTrailingSlash(new URL(rel, window.location.origin + '/').href)
  }
  // Fallback: origin + BASE_URL + 'api/'
  const baseUrl = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '/') // e.g. '/aiprototype/'
  return normalizeTrailingSlash(new URL('api/', window.location.origin + baseUrl).href)
}
const API_BASE = computeApiBase()
const api = axios.create({ baseURL: API_BASE, headers: { 'Content-Type': 'application/json' } })

/* ── State ──────────────────────────────────────────────────────────── */
const llmOptions = ref([])                // loaded from /api/models
const loadingModels = ref(false)

const model = ref('')                     // will be set from models API
const pendingModel = ref('')
const confirmModelChange = ref(false)

const draft = ref('')
const busy = ref(false)
const confirmReset = ref(false)

const snack = ref({ open: false, text: '' })

const messages = ref([
  { role: 'system', content: 'You are a helpful assistant.' },
])

const editBuffer = ref('')
const editingIndex = ref(-1)
const scrollEl = ref(null)

const userId = ref(localStorage.getItem('user_id') || generateUserId())
watch(userId, (v) => localStorage.setItem('user_id', v))

/* ── Labels based on loaded options ─────────────────────────────────── */
const modelLabel = computed(() => labelForValue(model.value) || (model.value || 'Model'))
const pendingModelLabel = computed(() => labelForValue(pendingModel.value) || (pendingModel.value || 'Model'))
const visibleMessages = computed(() => messages.value.filter(m => m.role !== 'system'))

onMounted(async () => {
  await loadModels()
  maybeScrollToBottom()
})

/* ── Helpers ────────────────────────────────────────────────────────── */
function labelForValue(val) {
  const opt = llmOptions.value.find(o => o.value === val)
  return opt ? opt.label : ''
}

function generateUserId() {
  const id = 'u_' + Math.random().toString(36).slice(2) + Date.now().toString(36)
  localStorage.setItem('user_id', id)
  return id
}

function copyMessage(text) {
  navigator.clipboard?.writeText(text)
  snack.value = { open: true, text: 'Copied to clipboard' }
}

function startEdit(i) {
  const msg = visibleMessages.value[i]
  if (msg?.role !== 'user') return
  visibleMessages.value.forEach(m => (m.editing = false))
  msg.editing = true
  editingIndex.value = i
  editBuffer.value = msg.content
}

function cancelEdit(i) {
  const msg = visibleMessages.value[i]
  if (!msg) return
  msg.editing = false
  editingIndex.value = -1
  editBuffer.value = ''
}

async function confirmEdit(i) {
  const msg = visibleMessages.value[i]
  if (!msg) return
  msg.content = editBuffer.value.trim() || msg.content
  msg.editing = false
  editingIndex.value = -1
  editBuffer.value = ''
  await nextTick()
  maybeScrollToBottom()
  // Per requirement: pressing OK sends a request
  await callLLM()
}

/* Build prompt array exactly as required */
function buildPromptArray() {
  return messages.value
    .filter(m => m.role !== 'system')
    .map(m => ({ role: m.role, prompts: m.content }))
}

async function sendDraft() {
  const text = draft.value.trim()
  if (!text || busy.value) return
  messages.value.push({ role: 'user', content: text })
  draft.value = ''
  await nextTick()
  maybeScrollToBottom()
  await callLLM()
}

async function callLLM() {
  if (!model.value) {
    snack.value = { open: true, text: 'Please select a model first.' }
    return
  }
  busy.value = true
  try {
    const prompt = buildPromptArray()
    const payload = { prompt, model: model.value }

    const { data } = await api.post('get-response', payload)
    const answer =
      (data && (data.answer || data.text || data.content)) ??
      (data?.choices?.[0]?.message?.content || '') ??
      'OK.'
    messages.value.push({ role: 'assistant', content: String(answer || 'OK.') })
    await nextTick()
    maybeScrollToBottom()
  } catch (e) {
    console.error(e)
    messages.value.push({ role: 'assistant', content: 'Sorry, something went wrong.' })
  } finally {
    busy.value = false
  }
}

function maybeScrollToBottom() {
  requestAnimationFrame(() => {
    const el = scrollEl.value || document.querySelector('.chat-scroll')
    if (el) el.scrollTop = el.scrollHeight
  })
}

function onModelPick() {
  if (pendingModel.value !== model.value) confirmModelChange.value = true
}
function cancelModelChange() {
  pendingModel.value = model.value
  confirmModelChange.value = false
}
function applyModelChange() {
  model.value = pendingModel.value
  confirmModelChange.value = false
  resetChat()
}

function resetChat() {
  messages.value = [{ role: 'system', content: 'You are a helpful assistant.' }]
  confirmReset.value = false
  draft.value = ''
  nextTick(maybeScrollToBottom)
}

/* Load models from /api/models and populate dropdown */
async function loadModels() {
  loadingModels.value = true
  try {
    const { data } = await api.get('models')
    const items = Array.isArray(data?.models) ? data.models : []
    if (!items.length) {
      llmOptions.value = [
        { label: 'GPT (fallback)', value: 'gpt' },
        { label: 'Llama (fallback)', value: 'llama' }
      ]
    } else {
      llmOptions.value = items.map(m => ({
        label: m.name || m.model_id || 'Model',
        value: m.model_id || '',
        id: m.id
      }))
    }
    if (!model.value && llmOptions.value.length) {
      model.value = llmOptions.value[0].value
      pendingModel.value = model.value
    }
  } catch (err) {
    console.error('Failed to load models:', err)
    snack.value = { open: true, text: 'Could not load models. Using fallback.' }
    llmOptions.value = [
      { label: 'GPT (fallback)', value: 'gpt' },
      { label: 'Llama (fallback)', value: 'llama' }
    ]
    if (!model.value) {
      model.value = llmOptions.value[0].value
      pendingModel.value = model.value
    }
  } finally {
    loadingModels.value = false
  }
}
</script>

<style scoped>
/* Fixed App Bar + content offset */
.fixed-bar { position: fixed !important; top: 0; left: 0; right: 0; z-index: 1000; }
.main-offset { padding-top: 80px; }

/* Light background */
.app-bg { background: #f3f4f6; min-height: 100vh; }

/* Global font (scoped-safe) */
:global(body) {
  font-family: 'Nunito', system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif !important;
}

/* Stronger titles/buttons */
.fw-700 { font-weight: 700 !important; }

/* Outline theme (blue-gray, 2px) */
.outlined-2 { border-color: #64748b !important; border-width: 2px !important; }

/* Uniform 6px radius everywhere */
.r6 { border-radius: 6px !important; }
:deep(.v-card),
:deep(.v-sheet),
:deep(.v-field),
:deep(.v-btn),
:deep(.v-chip),
:deep(.v-list),
:deep(.v-textarea),
:deep(.v-select) { border-radius: 6px !important; }

/* Keep Vuetify outlines tidy */
:deep(.v-field--variant-outlined) { --v-field-border-width: 2px; }
:deep(.v-field--variant-outlined .v-field__outline__start),
:deep(.v-field--variant-outlined .v-field__outline__end),
:deep(.v-field--variant-outlined .v-field__outline__notch) { border-color: #64748b !important; }

/* Appbar bottom line */
.appbar { border-bottom: 2px solid #64748b; }

/* User box sizing */
.user-box { min-width: 420px; }

/* Chat area container */
.chat-scroll {
  max-height: 52vh;
  overflow-y: auto;
  padding: 8px;
  background: rgba(0,0,0,0.02);
  border: 1px solid rgba(100,116,139,0.35);
}

/* Bubbles */
.bubble {
  line-height: 1.4;
  word-break: break-word;
  display: inline-block;
  min-width: 220px;
}
@media (max-width: 480px) {
  .bubble { min-width: 160px; }
}
.bubble-user { background: #eef2ff; color: #111827; }
.bubble-assistant { background: #ffffff; color: #111827; }

/* Icon-only actions */
.bubble-wrap { position: relative; max-width: 80%; }
.bubble-actions {
  position: absolute;
  top: 6px;
  right: 6px;
  display: flex;
  gap: 6px;
  opacity: 0;
  transition: opacity .15s ease;
  padding: 2px;
}
.bubble-wrap:hover .bubble-actions { opacity: 1; }
.icon-dark :deep(svg) { color: #4b5563 !important; }

/* Side alignment */
.user-side .bubble { margin-left: auto; }
.assistant-side .bubble { margin-right: auto; }

/* Center labels/icons on buttons */
.center-btn { display: inline-flex; align-items: center; justify-content: center; }

/* Make textarea outline correct */
.msg-textarea :deep(.v-field) { border-radius: 6px !important; }

/* Consistent icon button size */
.bubble-actions :deep(button.v-btn) {
  width: 28px;
  height: 28px;
  min-width: 28px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

/* Edit-mode action buttons */
.edit-actions {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 10px;
}
.edit-actions :deep(.v-btn) { height: 36px; }

/* ── POPUPS: force white interior with outlined border ─────────────── */
:deep(.v-overlay__content .v-card),
:deep(.v-overlay__content .v-sheet),
:deep(.v-overlay__content .v-list) {
  background-color: #ffffff !important;
  color: #111827 !important;
  border: 2px solid #64748b !important;
  border-radius: 6px !important;
}

/* Snackbar surface */
:deep(.v-snackbar .v-snackbar__wrapper) {
  background-color: #ffffff !important;
  color: #111827 !important;
  border: 2px solid #64748b !important;
  border-radius: 6px !important;
}
</style>
