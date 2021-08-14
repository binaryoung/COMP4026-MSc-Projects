<template>
  <boxoban :room="room" :topology="topology" :control="true" :trajectory="trajectory"/>
  <button @click="solve()">solve</button>
</template>

<script setup>
import {reactive} from 'vue'
import Boxoban from './components/Boxoban.vue'

const room = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 3, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 1, 3, 5, 0],
  [0, 0, 0, 0, 0, 1, 3, 3, 1, 0],
  [0, 1, 2, 1, 2, 1, 1, 2, 1, 0],
  [0, 0, 1, 1, 1, 1, 1, 2, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

const topology = [
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
  [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
  [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
  [0, 1, 2, 1, 2, 1, 1, 2, 1, 0],
  [0, 0, 1, 1, 1, 1, 1, 2, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

const expertTrajectory = [
  2, 1, 3, 1, 2, 2, 2, 2, 2, 3, 3, 0, 3, 1, 3, 3, 0, 0, 6, 0, 2, 1, 1, 0, 0, 0,
  2, 2, 1, 3, 0, 3, 1, 1, 3, 1, 1, 2, 2, 0, 3, 0, 3, 1, 2, 2, 1, 1, 2, 2,
]

const trajectory = reactive({
  actions: [],
})

function solve() {
  trajectory.actions = expertTrajectory
}
</script>