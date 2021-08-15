<template>
  <screen :room="room" ref="screen" />
</template>

<script setup>
import { reactive, ref, watch } from 'vue'
import { onKeyStroke } from '@vueuse/core'
import confetti from 'canvas-confetti'

import Environment from '../environment'
import Screen from './Screen.vue'

const props = defineProps({
  room: Array,
  topology: Array,
  control: {
    type: Boolean,
    default: true,
  },
  trajectory: {
    type: Object,
    default: {},
  },
})

const room = reactive(props.room)
const environment = new Environment(room, props.topology)
const screen = ref(null)

if (props.control == true) {
  onKeyStroke('ArrowUp', (e) => {
    e.preventDefault()
    step(0)
  })
  onKeyStroke('ArrowDown', (e) => {
    e.preventDefault()
    step(1)
  })
  onKeyStroke('ArrowLeft', (e) => {
    e.preventDefault()
    step(2)
  })
  onKeyStroke('ArrowRight', (e) => {
    e.preventDefault()
    step(3)
  })
}

watch(props.trajectory, async (value, _) => {
  for (let action of value.actions) {
    await new Promise((r) => setTimeout(r, 450))
    step(action)
  }
})

function step(action) {
  let result = environment.step(action)

  if (result.done == true && result.info.finished == true) {
    const { left, top, width, height } = screen.value.$el.getBoundingClientRect()

    confetti({
      particleCount: 250,
      spread: 80,
      ticks: 100,
      gravity: 1.5,
      origin: {
        x: (left + width / 2) / window.innerWidth,
        y: (top + height / 12 * 9) / window.innerHeight,
      },
    })
  }
}
</script>


