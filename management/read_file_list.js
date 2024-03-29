async function loadAndDisplayPosts(json_filename, id) {
    try {
        // JSON 파일 로드
        const response = await fetch(json_filename);
        const data = await response.json();

        // 날짜 기준으로 오름차순 정렬
        data.posts.sort((a, b) => a.time - b.date);

        // posts-container div에 게시글 링크 추가
        const container = document.getElementById(id);
        data.posts.forEach(post => {
            const anchor = document.createElement('a');
            anchor.href = post.link;
            anchor.textContent = `${post.title}`;
            anchor.style.display = 'block'; // 블록 스타일로 표시
            container.appendChild(anchor);
        });
        console.log('Success')
    } catch (error) {
        console.error('게시글을 로드하는 데 실패했습니다:', error);
    }
}