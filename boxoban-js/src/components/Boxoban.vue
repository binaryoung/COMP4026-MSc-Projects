<template>
  <screen :room="room" ref="screen"/>
</template>

<script setup>
import { reactive, ref, watch } from 'vue'
import { onKeyStroke, useElementBounding, useWindowSize } from '@vueuse/core'
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

const { x, y, top, right, bottom, left, width, height } = useElementBounding(screen)
const { windowWidth, windowHeight } = useWindowSize()

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
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
    })
  }
}
</script>


