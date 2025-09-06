<template>
  <v-app>
    <!-- Fixed App Bar -->
    <v-app-bar
      flat
      class="appbar outlined-2 r6 fixed-bar"
      height="64"
    >
      <v-toolbar-title class="fw-700">Chat Playground</v-toolbar-title>
      <v-spacer />
      <!-- Single outlined user box: "User: <id>" -->
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
            <!-- v-select with Font Awesome chevron via slot -->
            <v-select
              label="Select LLM"
              :items="llmOptions"
              item-title="label"
              item-value="value"
              v-model="pendingModel"
              variant="outlined"
              class="outlined-2 r6"
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
            <div class="chat-scroll r6 outlined-2-soft">
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
                    <div  style="white-space: pre-wrap;">
                      {{ m.content }}
                    </div>

                    <!-- Icon-only actions (Font Awesome component) -->
                    <div v-if="m.role === 'user'" class="bubble-actions">
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
                      <v-btn
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
                    />
                    <div class="edit-actions mt-2">
                      <v-btn
                        variant="outlined"
                        class="outlined-2 r6 fw-700 center-btn"
                        size="small"
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
                :disabled="!draft.trim()"
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

    <v-snackbar v-model="snack.open" :timeout="1800">
      {{ snack.text }}
    </v-snackbar>
  </v-app>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import axios from 'axios'

/* ── Font Awesome: local to this component ───────────────────────────── */
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
/* Now <FontAwesomeIcon icon="fa-solid fa-pen" /> works in this SFC only. */

/* ── App logic ───────────────────────────────────────────────────────── */
const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' }
})

const llmOptions = [
  { label: 'GPT',   value: 'gpt' },
  { label: 'Llama', value: 'llama' }
]

const model = ref('gpt')
const pendingModel = ref(model.value)
const confirmModelChange = ref(false)

const draft = ref('')
const busy = ref(false)
const confirmReset = ref(false)

const snack = ref({ open: false, text: '' })

const messages = ref([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is the capital of Austria?' },
  { role: 'assistant', content: 'Vienna is the capital of Austria.' },
  { role: 'user', content: 'Nice. Rough population of Vienna?' },
  { role: 'assistant', content: 'Approximately 1.9 million people.' }
])

const editBuffer = ref('')
const editingIndex = ref(-1)

const userId = ref(localStorage.getItem('user_id') || generateUserId())
watch(userId, (v) => localStorage.setItem('user_id', v))

const modelLabel = computed(() => (model.value === 'gpt' ? 'GPT' : 'Llama'))
const pendingModelLabel = computed(() => (pendingModel.value === 'gpt' ? 'GPT' : 'Llama'))
const visibleMessages = computed(() => messages.value.filter(m => m.role !== 'system'))

onMounted(() => {
  maybeScrollToBottom()
})

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
  if (msg.role !== 'user') return
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

function confirmEdit(i) {
  const msg = visibleMessages.value[i]
  if (!msg) return
  msg.content = editBuffer.value.trim() || msg.content
  msg.editing = false
  editingIndex.value = -1
  editBuffer.value = ''
}

async function sendDraft() {
  const text = draft.value.trim()
  if (!text) return
  messages.value.push({ role: 'user', content: text })
  draft.value = ''
  await callLLM()
}

async function callLLM() {
  busy.value = true
  try {
    const payload = {
      model: model.value,
      user_id: userId.value,
      messages: messages.value,
      temperature: 0.7,
      max_tokens: 512
    }
    const { data } = await api.post('/chat/completions', payload)
    const answer = data?.choices?.[0]?.message?.content || ''
    messages.value.push({ role: 'assistant', content: answer })
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
    const el = document.querySelector('.chat-scroll')
    if (el) el.scrollTop = el.scrollHeight
  })
}

function onModelPick() {
  if (pendingModel.value !== model.value) {
    confirmModelChange.value = true
  }
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
}
</script>

<style scoped>
/* Fixed App Bar + content offset */
.fixed-bar { position: fixed !important; top: 0; left: 0; right: 0; z-index: 1000; }
.main-offset { padding-top: 80px; }

/* Light background */
.app-bg { background: #f3f4f6; min-height: 100vh; }

/* Keep Nunito (as in your index.html) */
:global(html), :global(body), :global(#app), :global(.v-application) {
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

/* IMPORTANT: don't mess with field-notch parts; just color/width */
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
.bubble { line-height: 1.4; word-break: break-word; }
.bubble-user { background: #eef2ff; color: #111827; }
.bubble-assistant { background: #ffffff; color: #111827; }

/* Icon-only actions (no outlines) */
.bubble-wrap { position: relative; max-width: 80%; }
.bubble-actions {
  position: absolute;
  top: 6px;              /* was -10px */
  right: 6px;            /* was -6px */
  display: flex;
  gap: 6px;              /* a touch more space between icons */
  opacity: 0;
  transition: opacity .15s ease;
  padding: 2px;          /* keeps them off the exact edge */
}

.bubble-wrap:hover .bubble-actions { opacity: 1; }

.icon-dark :deep(svg) { color: #4b5563 !important; } /* color FA svgs */

/* Side alignment */
.user-side .bubble { margin-left: auto; }
.assistant-side .bubble { margin-right: auto; }

/* Center labels/icons on buttons */
.center-btn { display: inline-flex; align-items: center; justify-content: center; }

/* Make textarea outline correct (no odd shapes) */
.msg-textarea :deep(.v-field) { border-radius: 6px !important; }


/* Make the tiny icon buttons a consistent size & centered */
.bubble-actions :deep(button.v-btn) {
  width: 28px;
  height: 28px;
  min-width: 28px;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

/* Edit-mode action buttons: align nicely with equal spacing */
.edit-actions {
  display: flex;
  justify-content: flex-end;  /* right-aligned row */
  align-items: center;        /* vertical centering */
  gap: 10px;                  /* replaces mr-2 on OK */
}

/* Ensure the two action buttons have the same height for visual centering */
.edit-actions :deep(.v-btn) {
  height: 36px;               /* small but consistent */
}


</style>
