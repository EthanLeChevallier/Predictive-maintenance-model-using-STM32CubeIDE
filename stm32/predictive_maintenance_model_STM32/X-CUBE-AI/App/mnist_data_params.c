/**
  ******************************************************************************
  * @file    mnist_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-03-21T11:23:54+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "mnist_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_mnist_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_mnist_weights_array_u64[1015] = {
  0x3e109fd33ec51dd8U, 0x3e9d018c3e4bd90bU, 0x3f079d6e3f1ab2f4U, 0xbf2d161fbeed2708U,
  0x3f800cdfbe45d7e3U, 0xbf02a6a13ec3045bU, 0xbf3a5f5c3f3549d6U, 0x3f6c3a5fbec835d3U,
  0xbc3e2f5d3d5bb636U, 0xbf949a1cbd820659U, 0xbff56a31bff73558U, 0xbff1d3e3bff39f0aU,
  0xbfee3d95bff008bcU, 0xbfeaa746bfec726dU, 0xbfe710f8bfe8dc1fU, 0xbfe35ff6bfe545d1U,
  0xbfdfe45cbfe1af83U, 0xbfdc4e0ebfde1935U, 0xbfd8b7bfbfda82e7U, 0xbfd52171bfd6ec98U,
  0xbfd18b23bfd3564aU, 0xbfcdf4d5bfcfbffcU, 0xbfca5e87bfccfcb9U, 0xbfc6d013bfc7ed69U,
  0xbfc331eabfc4fd11U, 0xbfc0627abfc106b1U, 0xbfbc7f38bfbe6fe2U, 0xbfb86f00bfba654eU,
  0xbfb4d23dbfb5e262U, 0xbfb1c1e4bfb30d8bU, 0xbfad640cbfaf773cU, 0xbfaa15c7bfac17c9U,
  0xbfa688bcbfa84aa0U, 0xbfa205b5bfa4cc58U, 0xbf9f52ddbfa11e04U, 0xbf9b7a7abf9da543U,
  0xbf98354cbf9a9bbaU, 0xbf94c5dfbf965b19U, 0xbf90f9a4bf92c4cbU, 0xbf8d6356bf8f8f4bU,
  0xbf892ce4bf8bf714U, 0xbf867907bf881eedU, 0xbf82b49bbf854959U, 0xbf7d0c90bf80a920U,
  0xbf76919abf79f436U, 0xbf709bedbf73c033U, 0xbf68a33ebf6c1998U, 0xbf6191d9bf656322U,
  0xbf5ab72dbf5dddfeU, 0xbf5317b5bf563cbfU, 0xbf4b553fbf4e2b61U, 0xbf443fa7bf47bf1bU,
  0xbf3c92aebf408884U, 0xbf3479edbf38c581U, 0xbf2f0132bf323754U, 0xbf274f81bf2b6ccfU,
  0xbf2111debf2434acU, 0xbf19d92fbf1dc195U, 0xbf12a368bf15e824U, 0xbf0b2a25bf0ea88eU,
  0xbf04f495bf07fd37U, 0xbef9bcf5bf013dfdU, 0xbeeb8b06bef3dc87U, 0xbedb7668bee3726fU,
  0xbece4405bed5c0e9U, 0xbec094f9bec7b322U, 0xbeb0b8bebeb947e5U, 0xbea17b57bea923dbU,
  0xbe94313dbe9a8b9bU, 0xbe867227be8d3f55U, 0xbe6db21ebe7e8ad3U, 0xbe516f2bbe5f6bc5U,
  0xbe357cd5be44c580U, 0xbe19237abe27077eU, 0xbdf974c3be0bbf39U, 0xbdc25d87bddce28aU,
  0xbd8b654cbda8c1a9U, 0xbd263862bd5af24eU, 0xbc4a993dbcdacf2fU, 0x3c82727f3b0b47efU,
  0x3d35ad293cf72552U, 0x3d943b3d3d6f8809U, 0x3dce30543db077c0U, 0x3e01c9f63de8c0c3U,
  0x3e1df0633e102654U, 0x3e3c4aeb3e2ca15bU, 0x3e599abc3e4b827dU, 0x3e75f4d43e688739U,
  0x3e89381c3e822792U, 0x3e97acb53e8f9895U, 0x3ea697583e9f2020U, 0x3eb45d4b3eadc680U,
  0x3ec422b33ebce1aaU, 0x3ed16c243eca962eU, 0x3ee00e653ed844b3U, 0x3eee1c5f3ee73b74U,
  0x3efbaff53ef4bb1cU, 0x3f0606fd3f0229f5U, 0x3f0bc2b43f08e13aU, 0x3f13d3a23f0f24d7U,
  0x3f1b38e93f175201U, 0x3f21d2cb3f1d9790U, 0x3f2a772e3f2625f2U, 0x3f2fb67b3f2d0d1dU,
  0x3f377c293f33776fU, 0x3f3e36663f3af5adU, 0x3f4574783f41e617U, 0x3f4d39693f489142U,
  0x3f53e1d23f50e396U, 0x3f5a72cd3f577c73U, 0x3f628e443f5e88f0U, 0x3f69ac9f3f66520eU,
  0x3f6edbd63f6c1272U, 0x3f78b51e3f7455feU, 0x3f7f6e073f7d08c0U, 0x3f8314253f81a267U,
  0x3f878e8b3f850d9bU, 0x3f89affc3f887300U, 0x3f8dfecb3f8c564eU, 0x3f91acc73f9088acU,
  0x3f959ab33f92e2baU, 0x3f97e1d33f966f85U, 0x3f9cd0183f9a8cdcU, 0x3f9fee513f9eedcbU,
  0x3fa3ab8d3fa291f5U, 0x3fa811aa3fa54fc7U, 0x3faab13c3fa94764U, 0x3fae478a3fac7c63U,
  0x3fb1fac13fb06fd4U, 0x3fb574273fb3a8ffU, 0x3fb90a753fb73f4eU, 0x3fbc37ae3fbad59cU,
  0x3fc037113fbe6beaU, 0x3fc3cd5f3fc20238U, 0x3fc763ad3fc59886U, 0x3fcaf9fc3fc92ed5U,
  0x3fcde7083fccc523U, 0x3fd226983fd05b71U, 0x88898d8b92928b91U, 0x86837e8886918d91U,
  0x9088918690838d8cU, 0x828b8884908c9281U, 0x808282837c8e9092U, 0x8383838a83897d86U,
  0x8588868f8c978c8cU, 0x848c8b8b9386928aU, 0x877f888d8089888bU, 0x938088858e7e858bU,
  0x8c849182877e877dU, 0x858c7b86898c8e89U, 0x837f7d8788818088U, 0x898b808b88818081U,
  0x949190958f8d848dU, 0x7d878c7f818e8e7eU, 0x86828887857b8b7cU, 0x848892858a8c7f86U,
  0x7e878d8d89888f89U, 0x7e7e7e868b888287U, 0x7e807f808a8d8c83U, 0x8d828b8585858589U,
  0x8481898d8a8c868cU, 0x867d8a8182868084U, 0x83898a8984819389U, 0x8380897e8d897d86U,
  0x878c817f7f8d7b8cU, 0x8d908d8a7f7e8a7eU, 0x948487868f849185U, 0x87827d8b7c84898dU,
  0x808d7c897d838984U, 0x8d8a8c8b87838883U, 0x898180878680888eU, 0x81838e7f7f8b8d89U,
  0x8b89897e878a857cU, 0x8b7f898b8c8f8d88U, 0x7d7f8a887d858a86U, 0x887c828c8f807e82U,
  0x86948b868d868c7dU, 0x86858d8b958d7d8bU, 0x878b8b8b817c7f8dU, 0x8f85858e89848d80U,
  0x8d88928c858b848bU, 0x847e88888c89908aU, 0x8c967e7e92818d82U, 0x8e888e90858a8a90U,
  0x8a8d9793858d8e8cU, 0x8689868b89908c8aU, 0x858b8a8e8687928dU, 0x867a827e878c868bU,
  0x7695758d78857082U, 0x817f7057734b9b6bU, 0x8261848090878686U, 0x4b77605c696a7653U,
  0x746b5b6e4a686961U, 0x7b7d5f7361686659U, 0x6d619a7185778d86U, 0x806c81777c735c6eU,
  0x6b957f7195678368U, 0x8f7c8da370997894U, 0x9883748360839681U, 0x97878b96aa90947fU,
  0x6a8771848f978183U, 0x708771989b8987a2U, 0xb798ad88a4758b71U, 0x8c889d8cb595c892U,
  0xa28889c1a19a9e91U, 0xa3a594858b72629aU, 0xd78bb0a2a195c69dU, 0x768383749374ae7dU,
  0x9a898c86af838499U, 0x8396ac9ea2aeac93U, 0x9c7ab089c7849f7cU, 0x7ac68a8168977270U,
  0x90b5c3a3a59affadU, 0xaf7ca37b5c79809cU, 0x8f9b827ea8879a89U, 0x8392beb57cc39062U,
  0x536f757b77a980acU, 0x7e897a82a3888a92U, 0x8ca38d677f728e90U, 0x648545a160b294b4U,
  0x85839a819285636dU, 0x917e7e6d998a838fU, 0x7279649680968773U, 0x817467646875537eU,
  0x78819b8b908c947dU, 0x806a87628c95557bU, 0x8c7d8b774c928d7eU, 0x97739d65975b9f63U,
  0x79667c906f8e6e90U, 0xc59687a186978682U, 0xbb68e762d472fd9dU, 0xa3c185979f86ae77U,
  0x81b083837a94aeaeU, 0xc67fc787d37da4abU, 0xae95ba8ab394a877U, 0x818f979cc4ec8c95U,
  0x9c9087807f7a7b7cU, 0x9077a27f9faca999U, 0x9193989c939e8487U, 0x8ea1807d8a728585U,
  0xe1bbccd6afc09ac6U, 0xb4a39da0a894b6afU, 0x646a9d939ea9b9a0U, 0xbf8e929290a38977U,
  0x877e8782af84c688U, 0x8b7e8f977592738aU, 0x7f7b816b5f587b6bU, 0x5c715161665c7165U,
  0x63986caf4cba529eU, 0x733c60427b5c797eU, 0x695e715d814b7e46U, 0x81ae959c737b696cU,
  0x725d626d7d84909fU, 0x8e5779605120551fU, 0x9f89b977ab7e9e84U, 0x7e92768d9da6ad93U,
  0x7f3f9a7a84777b8dU, 0xc577b7978d77796eU, 0x6cd197afac7db17aU, 0xc08f9577849b71b5U,
  0x856a8e51723dd692U, 0xb083c56dc9779f7fU, 0x9486959f85b89bb7U, 0x9763c389c6879973U,
  0xd680758398888877U, 0xae8396a28e9faa86U, 0xa17ba45fa4779c88U, 0x8d7a8da3a1b1a782U,
  0x8cae88aea1819f6dU, 0x9b739f98968a8196U, 0xb7989b908c6a9656U, 0x9f67935e977e8e85U,
  0x84976e9b80987695U, 0x75777d7883898497U, 0x8c9e777bb7ae6f9fU, 0x5fa05a907c7aa96eU,
  0x7871798181906a92U, 0x95846a696a5e6554U, 0x6e8f8a8a8d868682U, 0x6d778e6f897d908dU,
  0x888c9f9490817a84U, 0x7f937d72967fb0a6U, 0xa59db89da9a186b1U, 0xaeb39fa1a1a09f9dU,
  0x7e856e768d9fa7b1U, 0x869f85988ba48d93U, 0x6b777d8775987f98U, 0x7e8c8a8f85897684U,
  0x8595899988a48596U, 0x6b5c65546b627575U, 0x9782757d8c698171U, 0x6483837f9c948587U,
  0x7563744d7d608ca3U, 0x937a827d7571716aU, 0x8a9c87948499838cU, 0x814191888d866fa0U,
  0xaa68ba6f83697c57U, 0x95999a7f957f9e73U, 0x9d9696b087a5829cU, 0xbc6ebe6a7665915dU,
  0xa16fb27dba85cf77U, 0x899f91a29599908cU, 0x9d918494b4ba8cb7U, 0xcf97bf81a975b264U,
  0x96928e76c77dd392U, 0xb2a09dae88a692a3U, 0xa5918270a7678396U, 0xa56fbea3b299c391U,
  0x8fa4899e9492a577U, 0x7e6b8fb7ad51af9bU, 0x9792b38da291838bU, 0x8c8e908aa372ac91U,
  0xd6678b897b936395U, 0x999d9d8180748b8eU, 0x9480d394cf8c9e84U, 0x6669627f7e8b968bU,
  0x7a748a8b87558679U, 0xcf8cb779aa78a979U, 0x8773a184c382d094U, 0x4851969089808879U,
  0xc476ab78746287acU, 0xae87c585b488c872U, 0x9da0918c8b8c8a8eU, 0x7e6686895561a782U,
  0xae8abb7bd677a17aU, 0x899186958da1a09cU, 0x638a8e7988858b7aU, 0x63666e5b826a8797U,
  0x6f976a9265804d7cU, 0x79789095779f8698U, 0x947086a37e7c5e7eU, 0x7c79445354298035U,
  0x858e6da16476657cU, 0x869a848c44446549U, 0x86798982927d8780U, 0x7c7492898a907a7bU,
  0x8d82857f707a7e72U, 0x86718c7882778878U, 0x747077437e4f8661U, 0x9e947680656c7e6eU,
  0x8177a36c92768a8eU, 0x6a768a6f8a7e8e72U, 0x5860505d4351896aU, 0x9399968a5e924c64U,
  0x967e8d897c4f7982U, 0xc053e851a47b8577U, 0x69697958cf5fa564U, 0x5748667c747b856cU,
  0xb69176a1a4d78bc1U, 0x9e51cf53d27be284U, 0x7b857f809872677cU, 0xa9d38fd58075a7b3U,
  0xab9f9f919f857fb6U, 0x81a66387a772d39bU, 0xaf92bea3aab474b4U, 0xc48c919facb688caU,
  0x9281f397bbb2a897U, 0x6c878ba5a4aa6aa6U, 0xdf9d8fcb8c747e84U, 0xed82dbacc7ab84a6U,
  0x9e7c837ea360c746U, 0x745a98938d7fa087U, 0xbcaec9a58eab8793U, 0x70748024c92ac994U,
  0xafb69ab091a08786U, 0x8cac8c87661a8e83U, 0x9622aa448d8189a6U, 0xa495a4999b902762U,
  0x795c6a5d969085a4U, 0x906492789a7a867eU, 0xa18d88897464a034U, 0x8756af84a0939592U,
  0x8d4f8572b0a64c3aU, 0xa8786787a263497cU, 0x9391968e8f888c87U, 0xa9a14d3a8c6ca595U,
  0x7fa6a08e8f71877aU, 0xa4908a889c8a9599U, 0x8d7a9996ab9c8887U, 0x8f7c85737d727741U,
  0x8d91857c61768c60U, 0x9da496947e9f8386U, 0x8d838d589a8eacb3U, 0x9e93897e85778980U,
  0x9bada9ab8fb29d94U, 0x8c92a4a0adb8b29eU, 0x868e807d827e8681U, 0x9d87809b86817283U,
  0x86aab78c8e91a38eU, 0x6f7eb3be74ae8eacU, 0x816689638484967aU, 0x8078967c997aa680U,
  0x568c6b80738d8388U, 0x81738b9fb2b98e87U, 0xb49aa19da28a8283U, 0x7da86eb279a38c9bU,
  0xcdd8759f6a9b72a1U, 0x71a19f708e4d9493U, 0x7abe8fb7989a8c9aU, 0x5e8a70916aaa6fbaU,
  0x9a6f8ca298927b8eU, 0x9791a6888d8a947dU, 0x70767598a7b78c9bU, 0x7f766f704c5c5f66U,
  0x9f787f7d923f8e96U, 0x987fb75fa27baa89U, 0x7e5b775880686c84U, 0x854b9080664da258U,
  0xa97f9f8c9f78506eU, 0x8c618d689669bb5bU, 0x7a4fb595b2789970U, 0x8b7783707a73904fU,
  0x994e7e60847a678eU, 0xa587a477a177a36cU, 0x8d908e81806cb186U, 0x8a598573947a9186U,
  0xa184927b8a5e9559U, 0x987fa08eb18c9d79U, 0x828677988489878eU, 0xa8a19cae82918e7fU,
  0x98a289968d8f9b9bU, 0x89909267867b9193U, 0x7cb25fa88b859d88U, 0x948e9da883be8fc0U,
  0x9973a87d8287907fU, 0x90ac7eb78ac08b73U, 0x8ba988a4b1ae8e98U, 0x868e84858c8f8c9dU,
  0x879885817492bda8U, 0x96c3b3b386b2937fU, 0x8481958b8c9f94b9U, 0x8f7cb4d2a4b2a9b2U,
  0x849190829185898aU, 0x8887868e8a828382U, 0x8782908688877d84U, 0x818f8f8c8890838eU,
  0x838c908188897e89U, 0x8389838988868380U, 0x8985858c817f888aU, 0x828b84868589908eU,
  0x86818c8b8f958f7fU, 0x83857d8b8c888f83U, 0x7e8a8b8291888e82U, 0x898a8e81818e8e8cU,
  0x8d858c8d7f869081U, 0x887d8d837e838f8cU, 0x92898b868881868eU, 0x87839082938a8a8dU,
  0x887e87828a898b81U, 0x8495828b7f8c8188U, 0x888486918a808191U, 0x8f8c8a8c807f8584U,
  0x8c877f89888d8988U, 0x8089808484958683U, 0x8c8b898b8c8a9188U, 0x8584868b8982958aU,
  0x8c8289828b8a848aU, 0x8c80878289868b82U, 0x8a8b8d84857d8d7eU, 0x86838c828b8d8681U,
  0x8f8287857d80888bU, 0x8c8d8e8c878c8287U, 0x87818c868886838eU, 0x8a89868b82868b87U,
  0x83827f828b8d8a85U, 0x8487858b88888481U, 0x8d8c8b878e858e80U, 0x87858989908d8c8eU,
  0x86847e888480828aU, 0x83908b84828e898cU, 0x8681839294868690U, 0x7e8c878984868a89U,
  0x8f8a8a8d8280887dU, 0x8790938a8b8f8b8cU, 0x82818b838689858eU, 0x85827d8286877e8cU,
  0x85858e8890818b89U, 0x858b8c8b8a888186U, 0x7d9085858e8d8889U, 0x8b89848a8c8d898dU,
  0x928f858e92847d7eU, 0x92a48b9a869c919bU, 0xa5a39caea2bfa6aeU, 0x98949ba5949b97acU,
  0x96b18ba18e928fa0U, 0xaea2999e929e828dU, 0x946f898aa480b1a9U, 0x84a78c8b8e92ac8aU,
  0x978e8f84778a919eU, 0xbf7ab573d2779c95U, 0xb686c18e9f80bb80U, 0x988f8691817fa37bU,
  0x8d6b88648a7c818eU, 0xa26ab072a66da06dU, 0x9184a3849a868380U, 0xa74fc0659792846dU,
  0x966f726b6b726074U, 0x9c808872a173aa75U, 0x99648c647f51726aU, 0x5e704d7672768086U,
  0x937aac86766c6362U, 0x6658665a9e8a967dU, 0x72887a668c4f89a1U, 0xa57c789659aa5aa0U,
  0x9097848a85709472U, 0x7f707fc96f857293U, 0x8a9c87aaaea9958aU, 0x958d8d88ae95a5acU,
  0x71b6a9c380b49392U, 0xa3b1b2897e9d8cbeU, 0xb7a9b8a1a387b4b1U, 0x7fb58db8a1b89c9cU,
  0xa2b28175aff389b2U, 0xa081ab8d8e987d8cU, 0x86ab8eaf96a48c8cU, 0xa6b87ea078b36bb4U,
  0x6b7e947f7fa885a0U, 0x699b769779908c89U, 0x819a80b574b173aeU, 0x879291b78f9d6187U,
  0xa081b89aad77868cU, 0x849e8082757d9c7bU, 0x93b84e667e6c959cU, 0x748c7b8a88598d9fU,
  0x677270707077767dU, 0x89655d6971717073U, 0x90828e9790843c3fU, 0x451e3d00531b855aU,
  0x550b502165344e3fU, 0x9799707e82705025U, 0x7c8b8a888a8d8894U, 0x707d899170a57784U,
  0x898974777a903e90U, 0x85a78f8888898d89U, 0x6e8490868d909eacU, 0x7e9d6f986d9d7994U,
  0x68866d797d6b6781U, 0x7da3848b95ab8a8fU, 0x86937f8f7398828cU, 0x7c727f7376847b8eU,
  0x81987d687d9e6c86U, 0x6990629668858497U, 0x978a8c959896808fU, 0x799b779d8b8d8f80U,
  0x7d95709687939091U, 0x7e91798476837093U, 0x9895a395a8abab9eU, 0x97c5898fa4ae94afU,
  0xa982b57f8c978992U, 0xa0a49999ae8b9098U, 0xa5829cab9ea39da6U, 0xdd859f8d987b8678U,
  0xc4a2baacb09ec184U, 0x878e7c826f999090U, 0x99788d808a40b485U, 0xa8b9c090bd80ab8aU,
  0x84946d829e8db2b4U, 0xc35ab5828d777d81U, 0x7d94a69ba390938dU, 0x696f6d7f77989383U,
  0xc26cad819d9a7189U, 0x74a89bab804dd187U, 0x8d8091888b90a0b1U, 0xad8e85915978786bU,
  0x5745d283ce96bf7dU, 0xb3908ba09ead9b85U, 0x7e897c8291909a85U, 0xb77ba36c9f788882U,
  0x96828381413d9c61U, 0x6f96878e78947f83U, 0x71687b7c6d9268a4U, 0x5e5b854fa062856fU,
  0x509982898d998793U, 0x809881ac61ac50a8U, 0x927378677073788aU, 0x899090887c796371U,
  0x9eac84b688b282aeU, 0x696975567e719389U, 0x94876e6a5a6c6b6cU, 0x879198978e8e918aU,
  0x829c8196757d7f8bU, 0x7d90837e8177888cU, 0x8f958a9493998784U, 0x714a6e2f6c588175U,
  0x6f68786371646568U, 0x9087685c6d676863U, 0x76678893848d9a8bU, 0x748c6b9c41805461U,
  0x7c705c755866746eU, 0x977f9ba78f726262U, 0x6bad7d938e8a9581U, 0x70a169a26eb263afU,
  0x837f7a7c6d87728fU, 0x75a6559597919aa2U, 0x63b172b38bb489a9U, 0x858392897fa072bcU,
  0x899b91a477928b78U, 0x80a08da99aa4599eU, 0x8592a7a185ba77adU, 0x8d9bae8aa98e9196U,
  0x75a984af7fba8fa0U, 0xa3a589aa8f9785abU, 0x928d899c9594ae97U, 0x888f939783b78da3U,
  0x7d86678c618c7594U, 0x8d96a19a958f7f83U, 0x65a2828c7a8a859fU, 0x8b7c69717683947dU,
  0x7d8a729194859a8bU, 0x7e8e899495927f8bU, 0x7b6a9094524f8a76U, 0x938599817f726669U,
  0x96818c81a28c9b9fU, 0x8aa1857999888788U, 0xa476716d7a539071U, 0x9b7b9f7cab71907dU,
  0x7e83a58b93729266U, 0x885d90808da3848bU, 0xc0729d74988e5f92U, 0x8c76a97da882ab6dU,
  0x8389718d8a8d9175U, 0xe48b8e8c908d8d88U, 0xc37ecb70b276b57dU, 0x997da17d9a72c367U,
  0x94b0939681716560U, 0xa07ea0a1b2a4a8a8U, 0xa28ca384aa7b967bU, 0x98876f4fad998299U,
  0x8b908e8085808688U, 0x9eac959f8a858b8aU, 0x82868e99838a9ab6U, 0x84868a8e84899485U,
  0x8b698c88838a868fU, 0x2e47243529377b41U, 0x7e86776d56665f57U, 0xa37595a3989c8881U,
  0x9969dc6586559d5fU, 0x817a857f8f649b63U, 0x8cc38a546f80737cU, 0x6b5e4c477d738d97U,
  0x926dae50bc4ab65bU, 0x6d738c86917e818aU, 0x9879c97e91948855U, 0xb7887a65894c4268U,
  0x87aaa0b8a2aa999aU, 0x7d908b879da28bb0U, 0x6a5685579264c077U, 0x9da8b7ada1c17e81U,
  0xc3d7b1da6ac895c6U, 0x7d7ab374796b8e78U, 0xabaa7da38e758f8fU, 0x70b35771a35e9b7fU,
  0x8e738b86dad777c3U, 0xbc78b98b97987f8dU, 0xa44d957c959476acU, 0x88a1595092678d46U,
  0x799972b982948e6dU, 0x8e92af92a4867477U, 0x7c719c65926f9078U, 0x88688972af8d7069U,
  0x7483549252977879U, 0x9161876c917e90a1U, 0xb1aea2856f5c9b5fU, 0x717d625880187e7cU,
  0x7c7a85917b898385U, 0x707cc2738a77816eU, 0x86778f84a0aca88bU, 0xb387a99bbb877584U,
  0x917f8f88a7859e84U, 0x969c9aac62958f86U, 0xa97f7a707c678d86U, 0x8fa79b9ec2a59f96U,
  0x8ca38d9596909d91U, 0x888a908998928e85U, 0x5b538a497042805bU, 0x96826a896e7a5b5dU,
  0x8a8e807297835a50U, 0x9384838d8d92868dU, 0x91628b84898c8985U, 0x9486849a968b774eU,
  0x918e8d868d8f8686U, 0x8e78967a8e709298U, 0xbdbdb4a6cd888e6aU, 0x5e687081a4acc0bfU,
  0xa576948796a58b89U, 0x9c87a485ae7cc47cU, 0x82ccadafb49ba890U, 0x6a98807a87a195bfU,
  0x9b8baa90a8859b73U, 0x8d979f859e819c82U, 0xc6c296bc6692809eU, 0xcb87d48451958188U,
  0x99809e77b47fd278U, 0x6f905a976c927487U, 0x91917f66b5c0729bU, 0xbf7cca7ec886df6cU,
  0x4c82727d9483ae88U, 0x87a4678a4d7a4384U, 0xa87aa8799d6a8c72U, 0x8cb5a1caaa99b286U,
  0x597a5f816d9080a6U, 0x8f8c897b705c6e76U, 0x94b37a948298808aU, 0x82958dae81c68fcaU,
  0x824b88697f72828dU, 0x6283708aaf889b86U, 0x90b569b7509f6372U, 0x9484977b9a909a9eU,
  0x74ab9da696569470U, 0x597b6c867186929fU, 0xa08e9c958990587eU, 0x845a9e87a084ac8aU,
  0x608d73a6989e8f94U, 0x879773857e8d598fU, 0xae86a490ab8fab8fU, 0x97b2885d9a82b78aU,
  0x7f8e74817c798082U, 0x9088a6929c8d908cU, 0x8c7c986a9489ab84U, 0x8a9b928f91a8839cU,
  0x9598a692c69ea7acU, 0x9693b28b8f9a9697U, 0x8a8d87838694bf93U, 0xa5bda9ad68748b78U,
  0x829480a27da289a8U, 0x8c92b7e76ea0758fU, 0x8381858488869283U, 0x8683908d8991888cU,
  0x868a818d87898387U, 0x8392858288818887U, 0x8d8a8c8d8b868a90U, 0x7e8488898b8b8581U,
  0x8e90838a928a848bU, 0x908a8c868783918fU, 0x82857d8d82838e85U, 0x8d7e8984867e8b8eU,
  0x8e83828c9090858bU, 0x888e8e8d918d868bU, 0x90837d7e86858a88U, 0x849186828283808bU,
  0x88928f8382889192U, 0x8686837f7d868882U, 0x838a898780898a86U, 0x82918990898a8782U,
  0x8d867f818583898dU, 0x818989817d7f8782U, 0x8a888d8d8b8d847eU, 0x8e84858d89888487U,
  0x82838b8e7d878c8bU, 0x8781878b838e7d83U, 0x8e828b9181898a84U, 0x8688848389808181U,
  0x8e837e8885838d8eU, 0x8d928c83838a7e89U, 0x8e888a8383878985U, 0x848684858c828289U,
  0x8d8581878885848cU, 0x8985858c828b8381U, 0x81858088897f8582U, 0x83828d868d828c83U,
  0x878684887d8b8881U, 0x85928888868e8a8dU, 0x80817e8c837f8282U, 0x828c8c83807d8c85U,
  0x928a928290858689U, 0x8280898384859191U, 0x8e82817e85838687U, 0x86849082838c7d7eU,
  0x88868c8a9092918eU, 0x848685888b8e8f86U, 0x8383848a8e8d8887U, 0x8e92888d8e839186U,
  0x8e928f9182858a84U, 0x8790848e878a838fU, 0x868689828a869288U, 0x8d858b807f868583U,
  0x81738868977b8d8dU, 0x86898b7d8f7e9680U, 0x82678385797e897dU, 0xa2897094817c765cU,
  0x8284688a6486a98eU, 0xa292c9aadfb85d8bU, 0x6e91836b7e647c81U, 0x7793749d639c7da2U,
  0x8785719a79916e8dU, 0x9067819ea8ab979eU, 0x8a96aa934c9f8681U, 0x8e8c889d86a49697U,
  0x98ac567a71737984U, 0xa694a0987f89777cU, 0x8c83a286af84d097U, 0x7a8d8e828f7d837cU,
  0x906c808094af8297U, 0xb786b890cf81959cU, 0x7e847978816a8a6fU, 0xa1b5a9a79f9d9d98U,
  0x9d9798759f7f818eU, 0x6d6e7c6b968d8fa5U, 0xb4a1a38283708b74U, 0x7c6f7c7aa0caa5a3U,
  0x8b9791a47e969988U, 0x9386878c7c83907bU, 0xa2bf6aa791a2b090U, 0xbc99927467607a97U,
  0x89989890bea4bfa1U, 0x7293949aa49f979fU, 0x9b6a7b6890cf5398U, 0xb4a9ca91b2938b83U,
  0x87a19eaaa399bfa5U, 0x7da8659e7ba98da2U, 0xce78ac7b79838295U, 0xa28aaa95b190bf7eU,
  0x76998092929c9a99U, 0x75918ba96a6d697bU, 0x838a847c9a74b872U, 0x7694898691858893U,
  0x71749e9c95aa77a0U, 0x7476957a85948385U, 0x76866e826189868eU, 0x8dc97ea967956f8dU,
  0x839688818f8a93c1U, 0x985b90789f7ba48fU, 0x796979648d678459U, 0x897e6e5a787f7b5dU,
  0x857e8776817a8d78U, 0x8aa4819b88718077U, 0x8671848189769b96U, 0x7e7680758b828a85U,
  0xb0b59ab69aaea57aU, 0x7d6f90a18daf9fc5U, 0xb097ae8b8e6d667aU, 0x8eb88eab969a8376U,
  0x64ac99c16cc786b1U, 0x755d7374839175a1U, 0x8aad8367a093745dU, 0x51bc4dba5aa36ea3U,
  0x8f8c949e8fb368b2U, 0x7484746aa26c9b71U, 0x35a469a8a4b78873U, 0x88ad73ab5fb92daeU,
  0xa471ac73ad7d998cU, 0x6692876d698aa589U, 0x609e569a45a26cbcU, 0xba8a868167946b8bU,
  0x95b0a5c591a3ab8eU, 0x45805e866a718356U, 0x89854b924a7c408bU, 0x61899783948b8c7dU,
  0xa571976dac99688fU, 0x547e4f6e376f5168U, 0x807b9290b0a266a2U, 0x839d4b8249727071U,
  0x9e7c8e979fb38bb6U, 0xb39a77ab67946081U, 0x6884637a797a8986U, 0xbcb8838290b27497U,
  0xa38da775ac7e87a9U, 0x8f89968e84907e9fU, 0xa1b391aa7a8f7b93U, 0x9e9a9a9c9da39388U,
  0x88946d88a085a995U, 0x97b296ae8baa94a8U, 0x92c788d8849589a0U, 0x5c967b92889496aaU,
  0x96a4859785947587U, 0x8c8da7bc90a19fa8U, 0x47a283818c7a8194U, 0x56905c9c37a84b87U,
  0x8ba493b79da37890U, 0x806880828e8e7daaU, 0x5e9b699b5caf617aU, 0x7aa38d957c9c759fU,
  0x8e75956a8d69a1a3U, 0x938e98928f979089U, 0x916b9578958c979dU, 0x8c8c949294818260U,
  0x9671939486979594U, 0x67aa8894848e9a9fU, 0x84817b889a8d878fU, 0x939e738c9e8e8377U,
  0x9db49aaa9db99091U, 0x9b839b9d83a29ca2U, 0xa278917181689a68U, 0xa2d29bb49b809e99U,
  0x839f91b0c2abdcb6U, 0xbf77bd7aa074898cU, 0x8b818b88b389b775U, 0x7eb06bb693df98caU,
  0xa7847083748e77a4U, 0xaf82c86fc976d382U, 0x91bf959b8072866aU, 0x6d8590877c825a92U,
  0xb28dca7e99797176U, 0x67598153c87bb482U, 0x95755d8089b392a9U, 0xaa8c9f7d9699a099U,
  0x8f716796a5a9b9acU, 0x89af88a833326644U, 0xa291a68baa786c78U, 0x92ae9db297a48b82U,
  0x7b4f805b919a7ea5U, 0x627e8b79928d8ca4U, 0x9aa0809993899282U, 0x8f857d80938f9298U,
  0x737090b277418e78U, 0x88846b844f7b6962U, 0x958f9a9096978e92U, 0x7c756f697e7e8891U,
  0x706a696a9a7e8b87U, 0x9f7b8f7993718871U, 0x7b6595798f7e9772U, 0x8e5a928d9c915452U,
  0xa579a87e72815778U, 0x957fa792ae8dbe82U, 0xa5b4566470648776U, 0xe5a4a6be94b48797U,
  0xd47fde79d17ad089U, 0x8e64857f987ab680U, 0x8ba9868f87905f66U, 0xc79bb2d0adc8a9b1U,
  0x9969a371a776b573U, 0x83906d50856b8777U, 0x3f5d0091bd9342f9U, 0xbf8086f1bf24025eU,
  0x3ea0b8a53ef9f927U, 0x3f1a7d20bd18e452U, 0x3e2fcfd7bf116cd3U, 0x3ebe91563f4aa3e3U,
  0xbee74e57bd75cf1cU, 0x3e8573ff3ed5a199U, 0xbde38eab3c76cb93U, 0x3e8f0e433e78334eU,
  0x3f05e91fbe2c9e38U, 0x3ea24835be2683f6U, 0xbf177aef3f1134fcU, 0xbfaf49a6bedbf715U,
  0x3e11905e3e6fc7afU, 0x3d13ceefbf08d535U, 0xbedb561e3e3953adU, 0xbe0f4b25be9d60e5U,
  0x3dea83293f07a990U, 0x3f31f6debe4c194dU, 0xbeb2b464bf916af1U, 0x3e5e60a03f51399dU,
  0xbfaee677bed9adceU, 0x3e32ff223eaba8ebU, 0x3cb1cc77bee0b24cU, 0x3d1adaa23d021305U,
  0x3dc08cc0bdbc7658U, 0x3f329cdd3ea20e13U, 0xbe13e88ebeee0483U, 0xbf479f0fbf8387b0U,
  0x3ec69e6ebe43f9d6U, 0xbe2c77a73f3456a7U, 0xbe9c0e023eb36fc6U, 0xbf7877903ed1e24aU,
  0xbe85c89abe83ed41U, 0xbdee1579bd9b400cU, 0xbe1074383edcbd65U, 0x3e67694bbee9f898U,
  0xbf2b84983e6d47a9U, 0x3e74718f3efac629U, 0x3eef2c22be5d8902U, 0x3eb799d3bef36a8bU,
  0xbfcd136fbdd6320cU, 0x3e1f0fcc3e750849U, 0xbf305406bd86e2ebU, 0x3eb2dd5ebec28236U,
  0x3ee9288fbd858c53U, 0x3ef54283bf8a618bU, 0x3dfb26c0be428dadU, 0xbea5cd3c3eabde47U,
  0x3ef079c4beaffbbaU, 0xbe7fb6eebea242daU, 0xbe894c6fbdbc6763U, 0x3f33f6b53f8a1fa5U,
  0xbe02e21a3e891b94U, 0xbf9bf6ebbe2bd13fU, 0xbfa1b2c8be6bcc74U, 0x3ebec42c3f4c2404U,
  0x3e571f89bda92917U, 0x3f27e333be196941U, 0xbf8d5e36be6e7da4U, 0x3e51435a3c11e1e4U,
  0x3e9e380fbd059edcU, 0xbdc4567fbf6621beU, 0x3e5776ee3d961df7U, 0xbf204d52befa3cebU,
  0xbc96e6da3f7d243aU, 0xbce4b213bd6adfe9U, 0x3f1c2dfabec8f3eaU, 0xbf13e872be8a2be6U,
  0x3ce694ff3ca3757dU, 0x3f04ff13be362717U, 0xbf0d9999be738870U, 0x3f3727923e7f5b33U,
  0xbf06ca83bf28348cU, 0xbe96a24dbefbfaf8U, 0x3d997ad03e15632dU, 0x3e2f6c15bedd2749U,
  0x3e0c2898bdc6e0caU, 0xbd57f9223e7a9ab5U, 0x3f01a2563e4bfa4cU, 0xbea0a84a3f288c0eU,
  0xbec2ccc0bf3dbe76U, 0xbf92b8ecbe9f6e2fU, 0x3e123b363ddc7fd5U, 0x3e1d4760bda61cfeU,
  0x3e6679393ea905bdU, 0x3dabef4abeb97d21U, 0x3f402784bf0ad9f2U, 0x3f16faea3ea2c1e5U,
  0x3e82621ebe2ee2abU, 0x3e27d9c2bf0a3039U, 0xbdcfcd81bf4ae30cU,
};


ai_handle g_mnist_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_mnist_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

