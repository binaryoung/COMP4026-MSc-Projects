<template>
  <div class="screen">
    <template v-for="(row, i) in room">
      <img v-for="(state, j) in row" :key="i + '-' + j" :src="render(state)" />
    </template>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

import wall from '../assets/wall.png'
import floor from '../assets/floor.png'
import target from '../assets/target.png'
import box from '../assets/box.png'
import boxOnTarget from '../assets/box_on_target.png'
import player from '../assets/player.png'
import playerOnTarget from '../assets/player_on_target.png'

defineProps({
  room: Array,
})

function render(state) {
  const map = {
    0: wall,
    1: floor,
    2: target,
    3: box,
    4: boxOnTarget,
    5: player,
    6: playerOnTarget,
  }

  return map[state]
}
</script>

<style scoped>
.screen {
  width: 230px;
  height: 230px;
  display: grid;
  grid-template-columns: repeat(10, 23px);
  grid-template-rows: repeat(10, 23px);
  grid-gap: 0px 0px;
}
img {
  height: 23px;
  widows: 23px;
}
</style>
